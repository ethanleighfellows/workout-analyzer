import { WorkoutSet } from '../types/workout';

export const calculateE1RM_Epley = (weight: number, reps: number): number => {
    if (reps === 1) return weight;
    // Epley formula: weight * (1 + reps/30)
    return Math.round(weight * (1 + reps / 30));
};

export const calculateE1RM_Brzycki = (weight: number, reps: number): number => {
    if (reps === 1) return weight;
    if (reps > 36) return calculateE1RM_Epley(weight, reps); // Brzycki breaks down at high reps
    // Brzycki formula: weight * (36 / (37 - reps))
    return Math.round(weight * (36 / (37 - reps)));
};

export const calculateIntensityPercentage = (reps: number): number => {
    // Rough estimate of %1RM based on reps performed
    const repPctMap: Record<number, number> = {
        1: 100, 2: 95, 3: 93, 4: 90, 5: 87,
        6: 85, 7: 83, 8: 80, 9: 77, 10: 75,
        11: 72, 12: 70, 15: 65, 20: 60
    };

    if (repPctMap[reps]) return repPctMap[reps];

    // Linear interpolation for missing values or extreme high reps
    if (reps > 20) return Math.max(0, 60 - (reps - 20) * 1.5);

    const closestLower = Object.keys(repPctMap).map(Number).filter(r => r < reps).pop() || 1;
    const closestHigher = Object.keys(repPctMap).map(Number).filter(r => r > reps).shift() || 20;

    const lowerVal = repPctMap[closestLower];
    const higherVal = repPctMap[closestHigher];
    const progress = (reps - closestLower) / (closestHigher - closestLower);

    return lowerVal - (progress * (lowerVal - higherVal));
};

export const calculateVolume = (weight: number, reps: number): number => {
    return weight * reps;
};

// 1. Fatigue & Readiness Index (ACWR) - Acute (7-day) to Chronic (28-day) Workload Ratio
export const calculateACWR = (workouts: WorkoutSet[]) => {
    if (!workouts.length) return [];

    // Group volume by date (YYYY-MM-DD)
    const dailyVolume: Record<string, number> = {};
    workouts.forEach(w => {
        const d = w.date.split(' ')[0];
        dailyVolume[d] = (dailyVolume[d] || 0) + (w.weight * w.reps);
    });

    const dates = Object.keys(dailyVolume).sort();
    if (!dates.length) return [];

    const startDate = new Date(dates[0]);
    const endDate = new Date(dates[dates.length - 1]);

    const results = [];

    for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
        const currentDateStr = d.toISOString().split('T')[0];

        // Calculate Acute (last 7 days, including today)
        let acuteVolume = 0;
        for (let i = 0; i < 7; i++) {
            const tempD = new Date(d);
            tempD.setDate(tempD.getDate() - i);
            acuteVolume += dailyVolume[tempD.toISOString().split('T')[0]] || 0;
        }

        // Calculate Chronic (last 28 days, including today)
        let chronicVolume = 0;
        for (let i = 0; i < 28; i++) {
            const tempD = new Date(d);
            tempD.setDate(tempD.getDate() - i);
            chronicVolume += dailyVolume[tempD.toISOString().split('T')[0]] || 0;
        }

        const chronicAverage = chronicVolume / 4; // 4 weeks
        const ratio = chronicAverage > 0 ? acuteVolume / chronicAverage : 0;

        // Only push data points where we actually had a workout or the ratio is meaningful
        if (dailyVolume[currentDateStr] || ratio > 0) {
            results.push({
                date: currentDateStr,
                acute: acuteVolume,
                chronic: chronicAverage,
                ratio: Number(ratio.toFixed(2))
            });
        }
    }
    return results;
};

// 2. Muscle Recovery Heatmap (Readiness Status based on 48h-72h window)
export const calculateRecoveryStatus = (workouts: WorkoutSet[]) => {
    const status: Record<string, { lastTrained: string, recoveryPct: number }> = {};
    const now = new Date();

    workouts.forEach(w => {
        if (!w.muscleGroup || w.muscleGroup === 'Other') return;
        const wDate = new Date(w.date.replace(' ', 'T'));

        if (!status[w.muscleGroup] || wDate > new Date(status[w.muscleGroup].lastTrained)) {
            const diffHours = (now.getTime() - wDate.getTime()) / (1000 * 60 * 60);

            // Assume 72 hours for 100% recovery for simplicity
            let pct = Math.min(100, Math.round((diffHours / 72) * 100));

            status[w.muscleGroup] = {
                lastTrained: w.date,
                recoveryPct: pct
            };
        }
    });

    return status;
};

// 3. Plateau / Sticking Point Detection
export const detectPlateaus = (workouts: WorkoutSet[]) => {
    const plateaus: { exercise: string, status: 'stalled' | 'progressing' }[] = [];

    // Group by exercise
    const exerciseData: Record<string, { date: string, e1rm: number }[]> = {};
    workouts.forEach(w => {
        if (w.e1rmBrzycki) {
            if (!exerciseData[w.exerciseName]) exerciseData[w.exerciseName] = [];
            exerciseData[w.exerciseName].push({ date: w.date.split(' ')[0], e1rm: w.e1rmBrzycki });
        }
    });

    const fourWeeksAgo = new Date();
    fourWeeksAgo.setDate(fourWeeksAgo.getDate() - 28);

    Object.entries(exerciseData).forEach(([exercise, data]) => {
        if (data.length < 5) return; // Need enough data

        // Find max in last 4 weeks vs max before 4 weeks
        const recentData = data.filter(d => new Date(d.date) >= fourWeeksAgo);
        const olderData = data.filter(d => new Date(d.date) < fourWeeksAgo);

        if (recentData.length > 0 && olderData.length > 0) {
            const recentMax = Math.max(...recentData.map(d => d.e1rm));
            const olderMax = Math.max(...olderData.map(d => d.e1rm));

            // If recent max isn't at least 2% better than older max, it's stalled
            if (recentMax <= olderMax * 1.02) {
                plateaus.push({ exercise, status: 'stalled' });
            } else {
                plateaus.push({ exercise, status: 'progressing' });
            }
        }
    });

    return plateaus;
};

// 4. Workout Density Metrics
export const calculateDensity = (workouts: WorkoutSet[]) => {
    // Group by Workout ID (Timestamp + Name)
    const sessions: Record<string, { volume: number, duration: number, date: string }> = {};

    workouts.forEach(w => {
        const id = `${w.date}_${w.workoutName}`;
        if (!sessions[id]) {
            sessions[id] = { volume: 0, duration: w.durationMinutes || 60, date: w.date.split(' ')[0] };
        }
        sessions[id].volume += (w.weight * w.reps);
    });

    return Object.values(sessions)
        .filter(s => s.duration > 0 && s.volume > 0)
        .map(s => ({
            date: s.date,
            density: Number((s.volume / s.duration).toFixed(2)) // lbs per minute
        }))
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
};

// 5. Mesocycle Tracking (Group by 7+ day gaps)
export const calculateMesocycles = (workouts: WorkoutSet[]) => {
    const dates = [...new Set(workouts.map(w => w.date.split(' ')[0]))].sort();
    const blocks: { start: string, end: string, count: number }[] = [];

    if (!dates.length) return blocks;

    let currentStart = dates[0];
    let lastDate = new Date(dates[0]);
    let count = 1;

    for (let i = 1; i < dates.length; i++) {
        const currDate = new Date(dates[i]);
        const diffDays = (currDate.getTime() - lastDate.getTime()) / (1000 * 60 * 60 * 24);

        if (diffDays >= 7) {
            // End of block
            blocks.push({ start: currentStart, end: lastDate.toISOString().split('T')[0], count });
            currentStart = dates[i];
            count = 1;
        } else {
            count++;
        }
        lastDate = currDate;
    }

    // Push the final block
    blocks.push({ start: currentStart, end: lastDate.toISOString().split('T')[0], count });

    return blocks;
};

// 6. Elite Archetype Comparisons (Legs vs Push vs Pull volume)
export const calculateArchetype = (workouts: WorkoutSet[]) => {
    let lower = 0, push = 0, pull = 0, isolation = 0;

    workouts.forEach(w => {
        const vol = w.weight * w.reps;
        if (w.exerciseCategory === 'lower_compound') lower += vol;
        else if (w.exerciseCategory === 'horizontal_push' || w.exerciseCategory === 'vertical_push') push += vol;
        else if (w.exerciseCategory === 'horizontal_pull' || w.exerciseCategory === 'vertical_pull') pull += vol;
        else isolation += vol;
    });

    const total = lower + push + pull + isolation;
    if (total === 0) return { lower: 0, push: 0, pull: 0, isolation: 0, archetype: 'Unknown' };

    const ratios = {
        lower: Number((lower / total).toFixed(2)),
        push: Number((push / total).toFixed(2)),
        pull: Number((pull / total).toFixed(2)),
        isolation: Number((isolation / total).toFixed(2))
    };

    let archetype = 'Balanced';
    if (ratios.lower > 0.45) archetype = 'Powerlifter (Lower Dominant)';
    else if (ratios.push > 0.4) archetype = 'Bro-Split (Push Dominant)';
    else if (ratios.isolation > 0.4) archetype = 'Bodybuilder (Isolation Heavy)';

    return { ...ratios, archetype };
};
