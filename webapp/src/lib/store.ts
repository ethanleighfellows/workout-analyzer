import { create } from 'zustand';
import { WorkoutSet, WorkoutSummary } from '../types/workout';
import { getAllWorkouts, hasStoredData, saveWorkouts } from './db';
import { calculateACWR, calculateRecoveryStatus, detectPlateaus, calculateDensity, calculateMesocycles, calculateArchetype } from './analytics';

export type DateFilter = 'all' | '1m' | '3m' | '6m' | '1y';

interface AppState {
    isLoaded: boolean;
    isLoading: boolean;
    workouts: WorkoutSet[];
    summary: WorkoutSummary | null;

    // Global Slicing
    dateFilter: DateFilter;
    filteredWorkouts: WorkoutSet[];
    filteredSummary: WorkoutSummary | null;

    error: string | null;

    // Actions
    initialize: () => Promise<void>;
    importData: (sets: WorkoutSet[]) => Promise<void>;
    clearData: () => Promise<void>;
    setDateFilter: (filter: DateFilter) => void;
}

const calculateSummary = (sets: WorkoutSet[]): WorkoutSummary | null => {
    if (sets.length === 0) return null;

    const dates = sets.map(s => new Date(s.date).getTime()).sort();
    const uniqueExercises = new Set(sets.map(s => s.exerciseName)).size;
    const totalVolume = sets.reduce((sum, set) => sum + (set.weight * set.reps), 0);
    const uniqueWorkouts = new Set(sets.map(s => s.date.split(' ')[0])).size;

    return {
        totalWorkouts: uniqueWorkouts,
        totalSets: sets.length,
        totalVolume,
        uniqueExercises,
        dateRange: {
            start: new Date(dates[0]).toISOString(),
            end: new Date(dates[dates.length - 1]).toISOString()
        },
        acwrData: calculateACWR(sets),
        recoveryStatus: calculateRecoveryStatus(sets),
        plateaus: detectPlateaus(sets),
        densityData: calculateDensity(sets),
        mesocycles: calculateMesocycles(sets),
        archetype: calculateArchetype(sets)
    };
};

const filterWorkoutsByDate = (workouts: WorkoutSet[], filter: DateFilter): WorkoutSet[] => {
    if (filter === 'all' || !workouts.length) return workouts;

    const now = new Date();
    let cutoff = new Date();
    if (filter === '1m') cutoff.setMonth(now.getMonth() - 1);
    if (filter === '3m') cutoff.setMonth(now.getMonth() - 3);
    if (filter === '6m') cutoff.setMonth(now.getMonth() - 6);
    if (filter === '1y') cutoff.setFullYear(now.getFullYear() - 1);

    return workouts.filter(w => new Date(w.date) >= cutoff);
};

export const useStore = create<AppState>((set, get) => ({
    isLoaded: false,
    isLoading: true,
    workouts: [],
    summary: null,
    dateFilter: 'all',
    filteredWorkouts: [],
    filteredSummary: null,
    error: null,

    initialize: async () => {
        try {
            set({ isLoading: true });
            const hasData = await hasStoredData();
            if (hasData) {
                const data = await getAllWorkouts();
                set({
                    workouts: data,
                    summary: calculateSummary(data),
                    filteredWorkouts: data,
                    filteredSummary: calculateSummary(data),
                    dateFilter: 'all',
                    isLoaded: true
                });
            }
        } catch (err: any) {
            set({ error: err.message });
        } finally {
            set({ isLoading: false });
        }
    },

    importData: async (sets: WorkoutSet[]) => {
        try {
            set({ isLoading: true, error: null });
            await saveWorkouts(sets);
            set({
                workouts: sets,
                summary: calculateSummary(sets),
                filteredWorkouts: sets,
                filteredSummary: calculateSummary(sets),
                dateFilter: 'all',
                isLoaded: true
            });
        } catch (err: any) {
            set({ error: `Failed to import data: ${err.message}` });
        } finally {
            set({ isLoading: false });
        }
    },

    clearData: async () => {
        set({ workouts: [], summary: null, filteredWorkouts: [], filteredSummary: null, isLoaded: false });
    },

    setDateFilter: (filter: DateFilter) => {
        const { workouts } = get();
        const filtered = filterWorkoutsByDate(workouts, filter);
        set({
            dateFilter: filter,
            filteredWorkouts: filtered,
            filteredSummary: calculateSummary(filtered)
        });
    }
}));
