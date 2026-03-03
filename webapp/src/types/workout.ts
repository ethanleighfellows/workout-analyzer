export interface WorkoutSet {
    id: string; // Unique identifier for the set
    date: string; // ISO 8601 string (YYYY-MM-DD HH:mm:ss)
    workoutName: string;
    durationMinutes: number;
    exerciseName: string;
    setOrder: number;
    weight: number;
    reps: number;
    distance?: number;
    seconds?: number;
    notes: string;
    workoutNotes: string;
    rpe?: number;

    // Computed fields (calculated during parsing/classification)
    e1rmEpley?: number;
    e1rmBrzycki?: number;
    intensityPct?: number;
    muscleGroup?: string;
    exerciseCategory?: string;
    isUnilateral?: boolean;
}

export interface WorkoutSummary {
    totalWorkouts: number;
    totalSets: number;
    totalVolume: number; // weight * reps
    uniqueExercises: number;
    dateRange: {
        start: string;
        end: string;
    };
    // Advanced Analytics Cache
    acwrData?: { date: string, acute: number, chronic: number, ratio: number }[];
    recoveryStatus?: Record<string, { lastTrained: string, recoveryPct: number }>;
    plateaus?: { exercise: string, status: 'stalled' | 'progressing' }[];
    densityData?: { date: string, density: number }[];
    mesocycles?: { start: string, end: string, count: number }[];
    archetype?: { lower: number, push: number, pull: number, isolation: number, archetype: string };
}
