#!/usr/bin/env python3
"""
Advanced Workout Progress Analyzer v2.0
========================================
Comprehensive workout analytics with advanced visualizations, muscle group tracking,
periodization analysis, and predictive insights.

Features:
- Interactive HTML dashboards with charts
- Exercise trajectory plotting
- Muscle group balance analysis
- Volume/intensity periodization tracking
- Training frequency patterns
- Fatigue and recovery scoring
- Performance predictions
- Comparative analysis across exercises

Usage:
    python workout_analyzer_v2.py analyze strong_workouts.csv
    python workout_analyzer_v2.py visualize strong_workouts.csv --output ./dashboard/
    python workout_analyzer_v2.py predict strong_workouts.csv --exercise "Bench Press (Dumbbell)"
"""

import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import warnings
import json
warnings.filterwarnings('ignore')

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

MUSCLE_GROUPS = {
    "Chest": {
        "exercises": [
            "Bench Press (Dumbbell)", "Incline Bench Press (Dumbbell)", 
            "Chest Fly", "Chest Press (Machine)", "Incline Chest Press (Machine)",
            "Chest Dip", "Cable Crossover", "Pec Deck (Machine)", 
            "Bench Press (Smith Machine)"
        ],
        "primary": True
    },
    "Back": {
        "exercises": [
            "Lat Pulldown (Cable)", "Row (Cable)", "Chest Supported Row (Machine)",
            "Pull Up", "Chin Up", "Seated Row (Cable)", "Seated Row (Machine)",
            "Lat Pulldown - Underhand (Cable)", "Pull Up (Assisted)",
            "Iso-Lateral Row (Machine)", "Lat Pulldown (Machine)"
        ],
        "primary": True
    },
    "Shoulders": {
        "exercises": [
            "Seated Overhead Press (Dumbbell)", "Front Raise (Plate)", 
            "Lateral Raise (Dumbbell)", "Rear Delt Fly (Machine)",
            "Face Pull (Cable)", "Reverse Fly (Machine)", 
            "Lateral Raise (Cable)", "Reverse Fly (Dumbbell)", "Y Raise"
        ],
        "primary": True
    },
    "Biceps": {
        "exercises": [
            "Bicep Curl (Dumbbell)", "Crush Curls", 
            "Reverse Grip Concentration Curl (Dumbbell)",
            "Hammer Curl (Dumbbell)", "Bicep Close Width Underhanded curl",
            "Preacher Curl (Machine)", "Bicep Curl (Cable)",
            "Bicep Curl (Barbell)", "Preacher Curl (Barbell)",
            "Preacher Curl (Dumbbell)", "Reverse Curl (Dumbbell)"
        ],
        "primary": False
    },
    "Triceps": {
        "exercises": [
            "Triceps Extension", "Triceps Extension (Machine)",
            "Triceps Extension (Cable)", "Triceps Extension (Dumbbell)",
            "Overhead Triceps Extension (Cable)", "Triceps Dip (Assisted)",
            "Triceps Pushdown (Cable - Straight Bar)", "Skullcrusher (Barbell)",
            "Tricep Press (Machine)", "Triceps Dip", "Triceps Extension (Barbell)"
        ],
        "primary": False
    },
    "Legs": {
        "exercises": [
            "Squat (Barbell)", "Leg Press", "Leg Extension (Machine)",
            "Leg Curl (Machine)", "Romanian Deadlift (Barbell)", "Hack Squat"
        ],
        "primary": True
    },
    "Calves": {
        "exercises": [
            "Standing Calf Raise (Machine)", "Seated Calf Raise (Plate Loaded)",
            "Calf Raise (Machine)"
        ],
        "primary": False
    },
    "Core": {
        "exercises": [
            "Crunch (Machine)", "Plank", "Hanging Leg Raise", "Sit Up"
        ],
        "primary": False
    }
}


EXERCISE_CATEGORIES = {
    # Pressing movements
    "Bench Press (Dumbbell)": {"category": "horizontal_push", "type": "main", "unilateral": False, "muscle": "Chest"},
    "Incline Bench Press (Dumbbell)": {"category": "incline_push", "type": "main", "unilateral": False, "muscle": "Chest"},
    "Chest Press (Machine)": {"category": "horizontal_push", "type": "main", "unilateral": False, "muscle": "Chest"},
    "Incline Chest Press (Machine)": {"category": "incline_push", "type": "main", "unilateral": False, "muscle": "Chest"},
    "Seated Overhead Press (Dumbbell)": {"category": "vertical_push", "type": "main", "unilateral": False, "muscle": "Shoulders"},
    "Chest Fly": {"category": "horizontal_push", "type": "accessory", "unilateral": False, "muscle": "Chest"},
    "Front Raise (Plate)": {"category": "vertical_push", "type": "accessory", "unilateral": False, "muscle": "Shoulders"},

    # Pulling movements
    "Lat Pulldown (Cable)": {"category": "vertical_pull", "type": "main", "unilateral": False, "muscle": "Back"},
    "Row (Cable)": {"category": "horizontal_pull", "type": "main", "unilateral": False, "muscle": "Back"},
    "Chest Supported Row (Machine)": {"category": "horizontal_pull", "type": "main", "unilateral": False, "muscle": "Back"},

    # Arms
    "Bicep Curl (Dumbbell)": {"category": "biceps", "type": "accessory", "unilateral": False, "muscle": "Biceps"},
    "Crush Curls": {"category": "biceps", "type": "accessory", "unilateral": False, "muscle": "Biceps"},
    "Reverse Grip Concentration Curl (Dumbbell)": {"category": "biceps", "type": "accessory", "unilateral": True, "muscle": "Biceps"},
    "Triceps Extension": {"category": "triceps", "type": "accessory", "unilateral": False, "muscle": "Triceps"},
    "Triceps Extension (Machine)": {"category": "triceps", "type": "accessory", "unilateral": False, "muscle": "Triceps"},
}




# ============================================================================
# INTELLIGENT EXERCISE CLASSIFIER
# ============================================================================

class ExerciseClassifier:
    """
    Multi-tier intelligent exercise categorization system.
    Automatically classifies exercises when not in MUSCLE_GROUPS dictionary.
    """

    def __init__(self):
        self.classification_rules = {
            'Chest': {
                'primary': ['bench press', 'chest press', 'chest fly', 'pec deck', 
                           'chest dip', 'push up', 'pushup', 'pec fly', 'crossover'],
                'secondary': ['cable fly', 'dumbbell fly', 'incline press', 
                             'decline press', 'flat press'],
                'equipment': ['pec', 'chest'],
            },
            'Back': {
                'primary': ['pulldown', 'pull down', 'row', 'pull up', 'pullup', 
                           'chin up', 'chinup', 'deadlift', 't-bar', 't bar'],
                'secondary': ['lat', 'shrug', 'pendlay', 'yates', 'meadows'],
                'equipment': ['lat', 'back'],
            },
            'Shoulders': {
                'primary': ['shoulder press', 'overhead press', 'military press',
                           'front raise', 'lateral raise', 'side raise', 'arnold press',
                           'upright row', 'bradford', 'landmine press'],
                'secondary': ['delt', 'ohp', 'viking press'],
                'equipment': ['shoulder', 'delt'],
            },
            'Biceps': {
                'primary': ['bicep curl', 'biceps curl', 'curl (dumbbell)', 
                           'curl (barbell)', 'curl (cable)', 'preacher curl',
                           'hammer curl', 'concentration curl', 'spider curl',
                           'ez bar curl', 'zottman curl', 'drag curl'],
                'secondary': ['21s', 'cheat curl'],
                'equipment': ['bicep', 'biceps'],
            },
            'Triceps': {
                'primary': ['tricep extension', 'triceps extension', 'skull crusher',
                           'skullcrusher', 'close grip bench', 'tricep pushdown',
                           'triceps pushdown', 'overhead extension', 'jm press',
                           'tate press', 'french press'],
                'secondary': ['tricep dip', 'triceps dip', 'kickback'],
                'equipment': ['tricep', 'triceps'],
            },
            'Legs': {
                'primary': ['squat', 'leg press', 'leg extension', 'leg curl',
                           'lunge', 'bulgarian split', 'hack squat', 'front squat',
                           'goblet squat', 'sissy squat', 'nordic curl', 'step up'],
                'secondary': ['quad', 'hamstring', 'glute', 'hip thrust', 
                             'glute bridge', 'sumo'],
                'equipment': ['leg', 'quad', 'hamstring', 'glute'],
            },
            'Calves': {
                'primary': ['calf raise', 'calf press', 'standing calf', 'seated calf'],
                'secondary': ['calves', 'gastrocnemius', 'soleus'],
                'equipment': ['calf'],
            },
            'Core': {
                'primary': ['crunch', 'sit up', 'situp', 'plank', 'ab wheel',
                           'leg raise', 'hanging leg', 'russian twist', 'woodchop',
                           'v up', 'v-up', 'deadbug', 'dead bug', 'flutter kick',
                           'bicycle', 'mountain climber', 'hollow hold'],
                'secondary': ['abs', 'oblique', 'ab crunch', 'cable crunch'],
                'equipment': ['ab', 'core'],
            },
            'Cardio': {
                'primary': ['running', 'treadmill', 'cycling', 'bike', 'rowing',
                           'elliptical', 'stair', 'stairs', 'walking', 'jogging',
                           'swimming', 'burpee', 'jump rope', 'jumping jack',
                           'box jump', 'battle rope'],
                'secondary': ['hiit', 'cardio', 'conditioning'],
                'equipment': [],
            }
        }

        self.compound_movements = {
            'deadlift': 'Back',
            'squat': 'Legs',
            'bench press': 'Chest',
            'overhead press': 'Shoulders',
            'row': 'Back',
            'pull up': 'Back',
            'dip': self._classify_dip,
        }

    def classify_exercise(self, exercise_name):
        """Main classification with multi-tier logic"""
        exercise_lower = exercise_name.lower().strip()

        # Tier 1: Compound movements
        for compound, result in self.compound_movements.items():
            if compound in exercise_lower:
                if callable(result):
                    return result(exercise_lower)
                return result

        # Tier 2: Primary keywords
        for muscle_group, rules in self.classification_rules.items():
            for keyword in rules['primary']:
                if keyword in exercise_lower:
                    return muscle_group

        # Tier 3: Secondary keywords
        for muscle_group, rules in self.classification_rules.items():
            for keyword in rules['secondary']:
                if keyword in exercise_lower:
                    return muscle_group

        # Tier 4: Equipment-based
        for muscle_group, rules in self.classification_rules.items():
            for keyword in rules['equipment']:
                if keyword in exercise_lower:
                    return muscle_group

        # Tier 5: Intelligent fallback
        fallback = self._fallback_classification(exercise_lower)
        if fallback:
            return fallback

        return 'Other'

    def _classify_dip(self, exercise_name):
        """Distinguish chest vs triceps dips"""
        if 'chest' in exercise_name:
            return 'Chest'
        elif 'tricep' in exercise_name or 'triceps' in exercise_name:
            return 'Triceps'
        return 'Triceps'

    def _fallback_classification(self, exercise_name):
        """Advanced biomechanical pattern matching"""
        # Press movements
        if 'press' in exercise_name:
            if any(w in exercise_name for w in ['leg', 'calf']):
                return 'Legs'
            elif any(w in exercise_name for w in ['shoulder', 'overhead', 'military', 'ohp']):
                return 'Shoulders'
            elif any(w in exercise_name for w in ['chest', 'bench', 'incline', 'decline', 'flat']):
                return 'Chest'
            elif 'close' in exercise_name or 'narrow' in exercise_name:
                return 'Triceps'
            elif any(w in exercise_name for w in ['french', 'jm', 'landmine']):
                return 'Triceps'

        # Extension movements
        if 'extension' in exercise_name:
            if 'leg' in exercise_name or 'knee' in exercise_name:
                return 'Legs'
            elif 'back' in exercise_name or 'hip' in exercise_name:
                return 'Back'
            return 'Triceps'

        # Curl movements
        if 'curl' in exercise_name:
            if 'leg' in exercise_name:
                return 'Legs'
            return 'Biceps'

        # Raise movements
        if 'raise' in exercise_name:
            if 'calf' in exercise_name:
                return 'Calves'
            elif 'leg' in exercise_name and ('hanging' in exercise_name or 'lying' in exercise_name):
                return 'Core'
            return 'Shoulders'

        # Fly movements
        if 'fly' in exercise_name or 'flye' in exercise_name:
            if 'rear' in exercise_name or 'reverse' in exercise_name:
                return 'Shoulders'
            return 'Chest'

        # Pulldown (always back)
        if 'pull' in exercise_name and ('down' in exercise_name or 'apart' in exercise_name):
            return 'Back'

        # Push movements
        if 'push' in exercise_name:
            if 'down' in exercise_name:
                return 'Triceps'
            elif 'up' in exercise_name:
                return 'Chest'

        return None

    def get_classification_confidence(self, exercise_name):
        """Return confidence: high, medium, low, unknown"""
        exercise_lower = exercise_name.lower().strip()

        # High: primary keywords or compounds
        for muscle_group, rules in self.classification_rules.items():
            if any(k in exercise_lower for k in rules['primary']):
                return 'high'

        if any(c in exercise_lower for c in self.compound_movements.keys()):
            return 'high'

        # Medium: secondary or equipment
        for muscle_group, rules in self.classification_rules.items():
            if any(k in exercise_lower for k in rules['secondary']):
                return 'medium'
            if any(k in exercise_lower for k in rules['equipment']):
                return 'medium'

        # Low: fallback rules
        if self._fallback_classification(exercise_lower):
            return 'low'

        return 'unknown'

# ============================================================================
# DATA MODELS
# ============================================================================

class WorkoutSet:
    """Enhanced workout set with additional metrics"""
    def __init__(self, date, exercise_name, set_order, weight, reps, notes="", rpe=None):
        self.date = pd.to_datetime(date)
        self.exercise_name = exercise_name
        self.set_order = set_order
        self.weight = float(weight) if pd.notna(weight) else 0.0
        self.reps = float(reps) if pd.notna(reps) else 0.0
        self.notes = str(notes) if pd.notna(notes) else ""
        self.rpe = float(rpe) if pd.notna(rpe) else None

        # Computed fields
        self.volume_load = self.weight * self.reps
        self.e1rm_epley = self._calc_e1rm_epley()
        self.e1rm_brzycki = self._calc_e1rm_brzycki()
        self.e1rm = self.e1rm_epley
        self.rir = self._estimate_rir_from_rpe()
        self.intensity_pct = self._calc_intensity_percentage()

    def _calc_e1rm_epley(self):
        if self.reps == 0 or self.weight == 0:
            return 0.0
        if self.reps == 1:
            return self.weight
        return self.weight * (1 + self.reps / 30.0)

    def _calc_e1rm_brzycki(self):
        if self.reps == 0 or self.weight == 0:
            return 0.0
        if self.reps == 1:
            return self.weight
        if self.reps >= 37:
            return self.weight * 2.0
        return self.weight / (1.0278 - 0.0278 * self.reps)

    def _estimate_rir_from_rpe(self):
        if self.rpe is not None:
            return max(0, 10 - self.rpe)
        return None

    def _calc_intensity_percentage(self):
        """Estimate %1RM based on reps"""
        if self.e1rm > 0:
            return (self.weight / self.e1rm) * 100
        return 0

    def confidence_level(self):
        if self.reps <= 5:
            return "high"
        elif self.reps <= 10:
            return "medium"
        else:
            return "low"


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_and_validate_csv(filepath, strict=False):
    """Load and validate workout CSV"""
    print(f"\n{'='*80}")
    print(f"LOADING DATA FROM: {filepath}")
    print(f"{'='*80}\n")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"ERROR: Could not read CSV file: {e}")
        sys.exit(1)

    required_cols = ['Date', 'Exercise Name', 'Weight', 'Reps']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df['Reps'] = pd.to_numeric(df['Reps'], errors='coerce')
    df['RPE'] = pd.to_numeric(df['RPE'], errors='coerce')

    initial_count = len(df)
    df = df[(df['Weight'] > 0) & (df['Reps'] > 0)].copy()
    removed_count = initial_count - len(df)

    if removed_count > 0:
        print(f"Ã¢Å¡ Ã¯Â¸Â  Removed {removed_count} rows with missing/invalid weight or reps")

    print(f"Ã¢Å“â€œ Loaded {len(df)} valid sets")
    print(f"Ã¢Å“â€œ Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Ã¢Å“â€œ Unique exercises: {df['Exercise Name'].nunique()}")
    print(f"Ã¢Å“â€œ Total workouts: {df.groupby(df['Date'].dt.date).ngroups}\n")

    return df


def compute_set_metrics(df):
    """Add computed metrics to each set"""
    sets = []

    for _, row in df.iterrows():
        workout_set = WorkoutSet(
            date=row['Date'],
            exercise_name=row['Exercise Name'],
            set_order=row.get('Set Order', ''),
            weight=row['Weight'],
            reps=row['Reps'],
            notes=row.get('Notes', ''),
            rpe=row.get('RPE')
        )
        sets.append(workout_set)

    df_computed = df.copy()
    df_computed['Volume_Load'] = [s.volume_load for s in sets]
    df_computed['E1RM_Epley'] = [s.e1rm_epley for s in sets]
    df_computed['E1RM_Brzycki'] = [s.e1rm_brzycki for s in sets]
    df_computed['E1RM'] = [s.e1rm for s in sets]
    df_computed['RIR_Estimated'] = [s.rir for s in sets]
    df_computed['Confidence'] = [s.confidence_level() for s in sets]
    df_computed['Intensity_Pct'] = [s.intensity_pct for s in sets]

    return df_computed


def add_muscle_group_tags(df):
    """Tag each exercise with muscle group using intelligent classification"""
    # First try hardcoded mappings
    muscle_map = {}
    for muscle, data in MUSCLE_GROUPS.items():
        for exercise in data['exercises']:
            muscle_map[exercise] = muscle

    # For unmapped exercises, use intelligent classifier
    unmapped_exercises = set(df['Exercise Name'].unique()) - set(muscle_map.keys())

    if unmapped_exercises:
        print(f"[INFO] Classifying {len(unmapped_exercises)} exercises intelligently...")
        classifier = ExerciseClassifier()

        confidence_counts = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}

        for exercise in unmapped_exercises:
            muscle_group = classifier.classify_exercise(exercise)
            confidence = classifier.get_classification_confidence(exercise)
            muscle_map[exercise] = muscle_group
            confidence_counts[confidence] += 1

        print(f"       High confidence:   {confidence_counts['high']} exercises")
        print(f"       Medium confidence: {confidence_counts['medium']} exercises")
        print(f"       Low confidence:    {confidence_counts['low']} exercises")
        if confidence_counts['unknown'] > 0:
            print(f"       [!] Unknown:       {confidence_counts['unknown']} exercises")

    df['Muscle_Group'] = df['Exercise Name'].map(muscle_map).fillna('Other')
    return df


def calculate_weekly_metrics(df):
    """Enhanced weekly aggregation"""
    df['Week'] = df['Date'].dt.to_period('W').dt.start_time
    df['Week_Number'] = (df['Date'] - df['Date'].min()).dt.days // 7

    weekly = df.groupby(['Week', 'Exercise Name']).agg({
        'Volume_Load': 'sum',
        'E1RM': 'max',
        'Reps': ['count', 'mean'],
        'Weight': ['max', 'mean'],
        'RPE': 'mean',
        'RIR_Estimated': 'mean',
        'Intensity_Pct': 'mean'
    }).reset_index()

    weekly.columns = ['Week', 'Exercise', 'Weekly_Tonnage', 'Top_E1RM', 
                      'Set_Count', 'Avg_Reps', 'Max_Weight', 'Avg_Weight',
                      'Avg_RPE', 'Avg_RIR', 'Avg_Intensity']

    return weekly


def calculate_muscle_group_weekly(df):
    """Weekly metrics by muscle group"""
    df['Week'] = df['Date'].dt.to_period('W').dt.start_time

    muscle_weekly = df.groupby(['Week', 'Muscle_Group']).agg({
        'Volume_Load': 'sum',
        'Reps': 'count',
        'E1RM': 'mean'
    }).reset_index()

    muscle_weekly.columns = ['Week', 'Muscle_Group', 'Total_Volume', 'Set_Count', 'Avg_E1RM']

    return muscle_weekly


# ============================================================================
# ADVANCED ANALYTICS
# ============================================================================

def calculate_training_frequency(df):
    """Analyze training frequency patterns"""
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Week'] = df['Date'].dt.to_period('W').dt.start_time

    # Sessions per week
    sessions_per_week = df.groupby('Week')['Date'].apply(
        lambda x: x.dt.date.nunique()
    ).reset_index()
    sessions_per_week.columns = ['Week', 'Sessions']

    # Day of week distribution
    day_distribution = df.groupby('Day_of_Week').size().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ], fill_value=0)

    # Average rest days
    workout_dates = sorted(df['Date'].dt.date.unique())
    rest_days = []
    for i in range(1, len(workout_dates)):
        rest_days.append((workout_dates[i] - workout_dates[i-1]).days - 1)

    return {
        'sessions_per_week': sessions_per_week,
        'day_distribution': day_distribution,
        'avg_rest_days': np.mean(rest_days) if rest_days else 0,
        'median_rest_days': np.median(rest_days) if rest_days else 0
    }


def calculate_fatigue_score(df, window_days=14):
    """Calculate rolling fatigue score based on volume and frequency"""
    df_sorted = df.sort_values('Date').copy()
    df_sorted['Week'] = df_sorted['Date'].dt.to_period('W').dt.start_time

    # Weekly volume
    weekly_volume = df_sorted.groupby('Week')['Volume_Load'].sum().reset_index()
    weekly_volume['Volume_MA'] = weekly_volume['Volume_Load'].rolling(window=4, min_periods=1).mean()
    weekly_volume['Volume_Ratio'] = weekly_volume['Volume_Load'] / weekly_volume['Volume_MA']

    # Fatigue score: volume ratio + frequency component
    weekly_volume['Fatigue_Score'] = weekly_volume['Volume_Ratio'].clip(0, 2) * 50

    return weekly_volume


def analyze_exercise_progression(df, exercise_name, window=12):
    """Detailed progression analysis for a specific exercise"""
    ex_data = df[df['Exercise Name'] == exercise_name].copy()

    if len(ex_data) == 0:
        return None

    ex_data = ex_data.sort_values('Date')

    # Rolling averages
    ex_data['E1RM_MA'] = ex_data['E1RM'].rolling(window=window, min_periods=1).mean()
    ex_data['Volume_MA'] = ex_data['Volume_Load'].rolling(window=window, min_periods=1).mean()
    ex_data['Weight_MA'] = ex_data['Weight'].rolling(window=window, min_periods=1).mean()

    # Best sets per week
    ex_data['Week'] = ex_data['Date'].dt.to_period('W').dt.start_time
    weekly_best = ex_data.groupby('Week').agg({
        'E1RM': 'max',
        'Volume_Load': 'sum',
        'Weight': 'max',
        'Reps': 'mean'
    }).reset_index()

    # Calculate rate of progression (linear regression on E1RM)
    if len(weekly_best) > 3:
        weeks_numeric = np.arange(len(weekly_best))
        coeffs = np.polyfit(weeks_numeric, weekly_best['E1RM'], 1)
        progression_rate = coeffs[0]  # lbs per week
    else:
        progression_rate = 0

    return {
        'data': ex_data,
        'weekly_best': weekly_best,
        'progression_rate': progression_rate,
        'current_e1rm': ex_data['E1RM'].iloc[-1],
        'pr_e1rm': ex_data['E1RM'].max(),
        'total_volume': ex_data['Volume_Load'].sum()
    }


def predict_future_performance(progression_data, weeks_ahead=4):
    """Predict future 1RM based on current progression rate"""
    if progression_data is None:
        return None

    current_e1rm = progression_data['current_e1rm']
    rate = progression_data['progression_rate']

    predictions = []
    for week in range(1, weeks_ahead + 1):
        predicted = current_e1rm + (rate * week)
        predictions.append({
            'weeks_ahead': week,
            'predicted_e1rm': predicted,
            'predicted_gain': predicted - current_e1rm
        })

    return pd.DataFrame(predictions)


def detect_volume_landmarks(weekly_df):
    """Detect significant volume milestones"""
    total_weekly = weekly_df.groupby('Week')['Weekly_Tonnage'].sum().reset_index()
    total_weekly = total_weekly.sort_values('Week')

    # Find all-time high weeks
    total_weekly['Is_Volume_PR'] = total_weekly['Weekly_Tonnage'] == total_weekly['Weekly_Tonnage'].cummax()

    volume_prs = total_weekly[total_weekly['Is_Volume_PR']].copy()

    return volume_prs


def analyze_muscle_balance(df):
    """Analyze training balance across muscle groups"""
    muscle_volumes = df.groupby('Muscle_Group')['Volume_Load'].sum().sort_values(ascending=False)
    muscle_sets = df.groupby('Muscle_Group').size().sort_values(ascending=False)

    # Calculate balance ratios for antagonist pairs
    balances = {}

    if 'Chest' in muscle_volumes and 'Back' in muscle_volumes:
        balances['Push/Pull (Chest/Back)'] = muscle_volumes['Chest'] / muscle_volumes['Back']

    if 'Biceps' in muscle_volumes and 'Triceps' in muscle_volumes:
        balances['Biceps/Triceps'] = muscle_volumes['Biceps'] / muscle_volumes['Triceps']

    return {
        'muscle_volumes': muscle_volumes,
        'muscle_sets': muscle_sets,
        'balance_ratios': balances
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_exercise_trajectory(progression_data, exercise_name, output_path):
    """Plot comprehensive exercise progression over time"""
    if progression_data is None:
        return

    data = progression_data['data']
    weekly = progression_data['weekly_best']

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. E1RM over time with trend
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(data['Date'], data['E1RM'], alpha=0.3, s=30, label='Individual Sets', color='steelblue')
    ax1.plot(data['Date'], data['E1RM_MA'], linewidth=2, label=f'Moving Average', color='darkred')
    ax1.plot(weekly['Week'], weekly['E1RM'], linewidth=2, marker='o', 
             label='Weekly Best', color='green', markersize=6)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Estimated 1RM (lbs)', fontsize=12)
    ax1.set_title(f'{exercise_name} - E1RM Progression', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Volume over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(weekly['Week'], weekly['Volume_Load'], linewidth=2, marker='s', 
             color='purple', markersize=5)
    ax2.set_xlabel('Week', fontsize=11)
    ax2.set_ylabel('Weekly Volume (lbs)', fontsize=11)
    ax2.set_title('Weekly Volume Load', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(weekly['Week'], weekly['Volume_Load'], alpha=0.3, color='purple')

    # 3. Weight progression
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(weekly['Week'], weekly['Weight'], linewidth=2, marker='D', 
             color='darkorange', markersize=5)
    ax3.set_xlabel('Week', fontsize=11)
    ax3.set_ylabel('Max Weight (lbs)', fontsize=11)
    ax3.set_title('Weekly Max Weight', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Rep distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(data['Reps'], bins=15, color='teal', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Reps', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Rep Range Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Stats summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    stats_text = f"""
    PROGRESSION STATISTICS
    Ã¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€Â

    Current E1RM:        {progression_data['current_e1rm']:.1f} lbs
    All-Time PR:         {progression_data['pr_e1rm']:.1f} lbs

    Progression Rate:    {progression_data['progression_rate']:.2f} lbs/week

    Total Volume:        {progression_data['total_volume']:,.0f} lbs
    Total Sets:          {len(data)}

    Avg Reps/Set:        {data['Reps'].mean():.1f}
    Avg Weight:          {data['Weight'].mean():.1f} lbs
    """

    ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.savefig(output_path / f'{exercise_name.replace(" ", "_")}_trajectory.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_muscle_group_distribution(df, muscle_weekly, output_path):
    """Visualize muscle group training distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Total volume by muscle group (pie chart)
    muscle_volumes = df.groupby('Muscle_Group')['Volume_Load'].sum().sort_values(ascending=False)
    colors = sns.color_palette('husl', len(muscle_volumes))

    axes[0, 0].pie(muscle_volumes, labels=muscle_volumes.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
    axes[0, 0].set_title('Total Volume Distribution by Muscle Group', fontsize=13, fontweight='bold')

    # 2. Volume over time by muscle group
    for muscle in muscle_weekly['Muscle_Group'].unique():
        muscle_data = muscle_weekly[muscle_weekly['Muscle_Group'] == muscle]
        axes[0, 1].plot(muscle_data['Week'], muscle_data['Total_Volume'], 
                       marker='o', label=muscle, linewidth=2)

    axes[0, 1].set_xlabel('Week', fontsize=11)
    axes[0, 1].set_ylabel('Weekly Volume (lbs)', fontsize=11)
    axes[0, 1].set_title('Volume Trends by Muscle Group', fontsize=13, fontweight='bold')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Set count by muscle group
    muscle_sets = df.groupby('Muscle_Group').size().sort_values(ascending=False)
    axes[1, 0].barh(muscle_sets.index, muscle_sets.values, color=colors)
    axes[1, 0].set_xlabel('Total Sets', fontsize=11)
    axes[1, 0].set_title('Set Count by Muscle Group', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. Weekly set count by muscle
    muscle_sets_weekly = muscle_weekly.pivot(index='Week', columns='Muscle_Group', 
                                              values='Set_Count').fillna(0)
    muscle_sets_weekly.plot(kind='area', stacked=True, ax=axes[1, 1], 
                           alpha=0.7, color=colors[:len(muscle_sets_weekly.columns)])
    axes[1, 1].set_xlabel('Week', fontsize=11)
    axes[1, 1].set_ylabel('Sets', fontsize=11)
    axes[1, 1].set_title('Weekly Set Distribution (Stacked)', fontsize=13, fontweight='bold')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'muscle_group_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_frequency(frequency_data, output_path):
    """Visualize training frequency patterns"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Sessions per week
    spw = frequency_data['sessions_per_week']
    axes[0].plot(spw['Week'], spw['Sessions'], marker='o', linewidth=2, 
                color='darkgreen', markersize=6)
    axes[0].axhline(spw['Sessions'].mean(), color='red', linestyle='--', 
                   label=f"Avg: {spw['Sessions'].mean():.1f}")
    axes[0].set_xlabel('Week', fontsize=11)
    axes[0].set_ylabel('Training Sessions', fontsize=11)
    axes[0].set_title('Weekly Training Frequency', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Day of week distribution
    day_dist = frequency_data['day_distribution']
    axes[1].bar(range(len(day_dist)), day_dist.values, color='steelblue', 
               edgecolor='black', alpha=0.8)
    axes[1].set_xticks(range(len(day_dist)))
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
    axes[1].set_ylabel('Total Sessions', fontsize=11)
    axes[1].set_title('Training by Day of Week', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # 3. Rest day stats
    axes[2].axis('off')
    rest_text = f"""
    REST & RECOVERY STATS
    Ã¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€Â

    Average Rest Days:     {frequency_data['avg_rest_days']:.1f}
    Median Rest Days:      {frequency_data['median_rest_days']:.1f}

    Avg Sessions/Week:     {spw['Sessions'].mean():.1f}
    Max Sessions/Week:     {spw['Sessions'].max()}
    Min Sessions/Week:     {spw['Sessions'].min()}
    """
    axes[2].text(0.1, 0.5, rest_text, fontsize=12, family='monospace',
                verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path / 'training_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_periodization_overview(weekly_df, fatigue_data, output_path):
    """Visualize periodization: volume, intensity, and fatigue cycles"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Aggregate by week
    weekly_totals = weekly_df.groupby('Week').agg({
        'Weekly_Tonnage': 'sum',
        'Avg_Intensity': 'mean',
        'Set_Count': 'sum'
    }).reset_index()

    # 1. Total weekly volume
    axes[0].bar(weekly_totals['Week'], weekly_totals['Weekly_Tonnage'], 
               color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].plot(weekly_totals['Week'], weekly_totals['Weekly_Tonnage'].rolling(4).mean(),
                color='darkred', linewidth=3, label='4-Week MA')
    axes[0].set_ylabel('Total Volume (lbs)', fontsize=12)
    axes[0].set_title('Weekly Volume Load - Periodization Overview', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Average intensity
    axes[1].plot(weekly_totals['Week'], weekly_totals['Avg_Intensity'], 
                marker='o', linewidth=2, color='darkorange', markersize=5)
    axes[1].axhline(weekly_totals['Avg_Intensity'].mean(), color='red', 
                   linestyle='--', label=f"Avg: {weekly_totals['Avg_Intensity'].mean():.1f}%")
    axes[1].set_ylabel('Avg Intensity (%1RM)', fontsize=12)
    axes[1].set_title('Weekly Average Training Intensity', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Fatigue score
    axes[2].plot(fatigue_data['Week'], fatigue_data['Fatigue_Score'], 
                marker='s', linewidth=2, color='darkred', markersize=5)
    axes[2].axhline(50, color='orange', linestyle='--', label='Baseline')
    axes[2].axhline(75, color='red', linestyle='--', label='High Fatigue')
    axes[2].fill_between(fatigue_data['Week'], 50, fatigue_data['Fatigue_Score'],
                        where=(fatigue_data['Fatigue_Score'] > 50), 
                        color='red', alpha=0.2)
    axes[2].set_xlabel('Week', fontsize=12)
    axes[2].set_ylabel('Fatigue Score', fontsize=12)
    axes[2].set_title('Estimated Fatigue Score', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'periodization_overview.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_exercises_comparison(df, top_n=6, output_path=None):
    """Compare top exercises by volume"""
    exercise_volumes = df.groupby('Exercise Name')['Volume_Load'].sum().sort_values(ascending=False).head(top_n)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (exercise, _) in enumerate(exercise_volumes.items()):
        ex_data = df[df['Exercise Name'] == exercise].copy()
        ex_data = ex_data.sort_values('Date')
        ex_data['E1RM_MA'] = ex_data['E1RM'].rolling(window=10, min_periods=1).mean()

        axes[idx].scatter(ex_data['Date'], ex_data['E1RM'], alpha=0.4, s=20)
        axes[idx].plot(ex_data['Date'], ex_data['E1RM_MA'], linewidth=2, color='darkred')
        axes[idx].set_title(exercise, fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Date', fontsize=9)
        axes[idx].set_ylabel('E1RM (lbs)', fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Top 6 Exercises - E1RM Progression Comparison', 
                fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path / 'top_exercises_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# REPORTING
# ============================================================================

def generate_html_dashboard(df, weekly_df, muscle_weekly, frequency_data, 
                           fatigue_data, output_path):
    """Generate comprehensive HTML report"""

    # Calculate summary statistics
    total_volume = df['Volume_Load'].sum()
    total_sets = len(df)
    unique_exercises = df['Exercise Name'].nunique()
    training_days = df['Date'].dt.date.nunique()
    date_range = f"{df['Date'].min().date()} to {df['Date'].max().date()}"

    muscle_balance = analyze_muscle_balance(df)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workout Analytics Dashboard</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .stat-value {{
                font-size: 32px;
                font-weight: bold;
                color: #667eea;
            }}
            .stat-label {{
                font-size: 14px;
                color: #666;
                margin-top: 5px;
            }}
            .section {{
                background: white;
                padding: 25px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h2 {{
                color: #333;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .chart-container img {{
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #667eea;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Ã°Å¸Ââ€¹Ã¯Â¸Â WORKOUT ANALYTICS DASHBOARD</h1>
            <p>{date_range}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_volume:,.0f}</div>
                <div class="stat-label">Total Volume (lbs)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_sets:,}</div>
                <div class="stat-label">Total Sets</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{training_days}</div>
                <div class="stat-label">Training Days</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{unique_exercises}</div>
                <div class="stat-label">Unique Exercises</div>
            </div>
        </div>

        <div class="section">
            <h2>Ã°Å¸â€œË† Periodization Overview</h2>
            <div class="chart-container">
                <img src="periodization_overview.png" alt="Periodization">
            </div>
        </div>

        <div class="section">
            <h2>Ã°Å¸â€™Âª Muscle Group Analysis</h2>
            <div class="chart-container">
                <img src="muscle_group_analysis.png" alt="Muscle Groups">
            </div>
        </div>

        <div class="section">
            <h2>Ã°Å¸â€œâ€¦ Training Frequency</h2>
            <div class="chart-container">
                <img src="training_frequency.png" alt="Frequency">
            </div>
        </div>

        <div class="section">
            <h2>Ã°Å¸Ââ€  Top Exercises Comparison</h2>
            <div class="chart-container">
                <img src="top_exercises_comparison.png" alt="Top Exercises">
            </div>
        </div>

        <div class="section">
            <h2>Ã¢Å¡â€“Ã¯Â¸Â Muscle Balance Ratios</h2>
            <table>
                <tr>
                    <th>Comparison</th>
                    <th>Ratio</th>
                    <th>Status</th>
                </tr>
    """

    for comparison, ratio in muscle_balance['balance_ratios'].items():
        status = "Ã¢Å“â€¦ Balanced" if 0.8 <= ratio <= 1.2 else "Ã¢Å¡ Ã¯Â¸Â Imbalanced"
        html_content += f"""
                <tr>
                    <td>{comparison}</td>
                    <td>{ratio:.2f}:1</td>
                    <td>{status}</td>
                </tr>
        """

    html_content += """
            </table>
        </div>

        <div class="section">
            <p style="text-align: center; color: #666; font-size: 12px;">
                Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
            </p>
        </div>
    </body>
    </html>
    """

    with open(output_path / 'dashboard.html', 'w') as f:
        f.write(html_content)

    print(f"Ã¢Å“â€¦ HTML Dashboard created: {output_path / 'dashboard.html'}")


# ============================================================================
# CLI COMMANDS
# ============================================================================

def cmd_analyze(args):
    """Enhanced analyze command"""
    df = load_and_validate_csv(args.csv_file)
    df = compute_set_metrics(df)
    df = add_muscle_group_tags(df)

    print("\n" + "="*80)
    print("COMPREHENSIVE WORKOUT ANALYSIS")
    print("="*80 + "\n")

    # Basic stats
    print(f"Ã°Å¸â€œÅ  SUMMARY STATISTICS")
    print(f"{'Ã¢â€â‚¬'*80}")
    print(f"Total Volume:        {df['Volume_Load'].sum():,.0f} lbs")
    print(f"Total Sets:          {len(df):,}")
    print(f"Unique Exercises:    {df['Exercise Name'].nunique()}")
    print(f"Training Days:       {df['Date'].dt.date.nunique()}")
    print(f"Date Range:          {df['Date'].min().date()} to {df['Date'].max().date()}")

    # Frequency analysis
    freq_data = calculate_training_frequency(df)
    print(f"\nAvg Sessions/Week:   {freq_data['sessions_per_week']['Sessions'].mean():.1f}")
    print(f"Avg Rest Days:       {freq_data['avg_rest_days']:.1f}\n")

    # Muscle group breakdown
    print(f"\nÃ°Å¸â€™Âª MUSCLE GROUP VOLUME")
    print(f"{'Ã¢â€â‚¬'*80}")
    muscle_volumes = df.groupby('Muscle_Group')['Volume_Load'].sum().sort_values(ascending=False)
    for muscle, volume in muscle_volumes.items():
        pct = (volume / muscle_volumes.sum()) * 100
        print(f"{muscle:15} {volume:12,.0f} lbs  ({pct:5.1f}%)")

    # Top exercises
    print(f"\nÃ°Å¸Ââ€  TOP 10 EXERCISES BY VOLUME")
    print(f"{'Ã¢â€â‚¬'*80}")
    top_exercises = df.groupby('Exercise Name')['Volume_Load'].sum().sort_values(ascending=False).head(10)
    for idx, (exercise, volume) in enumerate(top_exercises.items(), 1):
        print(f"{idx:2}. {exercise:45} {volume:12,.0f} lbs")

    print(f"\n{'='*80}\n")


def cmd_visualize(args):
    """Generate all visualizations"""
    df = load_and_validate_csv(args.csv_file)
    df = compute_set_metrics(df)
    df = add_muscle_group_tags(df)

    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)

    print("\nÃ°Å¸Å½Â¨ GENERATING VISUALIZATIONS...")
    print(f"{'='*80}\n")

    # Calculate all metrics
    weekly_df = calculate_weekly_metrics(df)
    muscle_weekly = calculate_muscle_group_weekly(df)
    frequency_data = calculate_training_frequency(df)
    fatigue_data = calculate_fatigue_score(df)

    # Generate plots
    print("Ã°Å¸â€œÅ  Creating periodization overview...")
    plot_periodization_overview(weekly_df, fatigue_data, output_path)

    print("Ã°Å¸â€™Âª Creating muscle group analysis...")
    plot_muscle_group_distribution(df, muscle_weekly, output_path)

    print("Ã°Å¸â€œâ€¦ Creating training frequency analysis...")
    plot_training_frequency(frequency_data, output_path)

    print("Ã°Å¸Ââ€  Creating top exercises comparison...")
    plot_top_exercises_comparison(df, top_n=6, output_path=output_path)

    # Generate exercise-specific plots for top exercises
    top_exercises = df.groupby('Exercise Name')['Volume_Load'].sum().sort_values(ascending=False).head(8)

    print(f"\nÃ°Å¸Å½Â¯ Creating detailed trajectories for top {len(top_exercises)} exercises...")
    for exercise in top_exercises.index:
        print(f"   - {exercise}")
        prog_data = analyze_exercise_progression(df, exercise)
        if prog_data:
            plot_exercise_trajectory(prog_data, exercise, output_path)

    # Generate HTML dashboard
    print("\nÃ°Å¸Å’Â Creating HTML dashboard...")
    generate_html_dashboard(df, weekly_df, muscle_weekly, frequency_data, 
                           fatigue_data, output_path)

    print(f"\n{'='*80}")
    print(f"Ã¢Å“â€¦ ALL VISUALIZATIONS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"Open dashboard.html in your browser to view the full report.\n")


def cmd_predict(args):
    """Predict future performance for an exercise"""
    df = load_and_validate_csv(args.csv_file)
    df = compute_set_metrics(df)

    exercise = args.exercise

    print(f"\n{'='*80}")
    print(f"PERFORMANCE PREDICTION: {exercise}")
    print(f"{'='*80}\n")

    prog_data = analyze_exercise_progression(df, exercise)

    if prog_data is None:
        print(f"Ã¢ÂÅ’ No data found for exercise: {exercise}")
        return

    predictions = predict_future_performance(prog_data, weeks_ahead=args.weeks)

    print(f"Current E1RM:        {prog_data['current_e1rm']:.1f} lbs")
    print(f"All-Time PR:         {prog_data['pr_e1rm']:.1f} lbs")
    print(f"Progression Rate:    {prog_data['progression_rate']:.2f} lbs/week\n")

    print("PREDICTED E1RM:")
    print(f"{'Ã¢â€â‚¬'*60}")
    for _, pred in predictions.iterrows():
        weeks = int(pred['weeks_ahead'])
        e1rm = pred['predicted_e1rm']
        gain = pred['predicted_gain']
        print(f"{weeks} week{'s' if weeks > 1 else '':1}:  {e1rm:6.1f} lbs  (+{gain:5.1f} lbs)")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Workout Progress Analyzer v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze workout data')
    analyze_parser.add_argument('csv_file', help='Path to workout CSV')

    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate all visualizations')
    viz_parser.add_argument('csv_file', help='Path to workout CSV')
    viz_parser.add_argument('--output', default='./dashboard', 
                           help='Output directory (default: ./dashboard)')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict future performance')
    predict_parser.add_argument('csv_file', help='Path to workout CSV')
    predict_parser.add_argument('--exercise', required=True, 
                               help='Exercise name to predict')
    predict_parser.add_argument('--weeks', type=int, default=4,
                               help='Weeks ahead to predict (default: 4)')

    args = parser.parse_args()

    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    elif args.command == 'predict':
        cmd_predict(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
