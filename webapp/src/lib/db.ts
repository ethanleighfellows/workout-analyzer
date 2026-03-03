import { openDB, DBSchema, IDBPDatabase } from 'idb';
import { WorkoutSet } from '../types/workout';

interface WorkoutDB extends DBSchema {
    workouts: {
        key: string; // Using ID string as key
        value: WorkoutSet;
        indexes: {
            'by-date': string;
            'by-exercise': string;
            'by-muscle': string;
        };
    };
    metadata: {
        key: string;
        value: any;
    };
}

const DB_NAME = 'ethan_workout_analyzer';
const DB_VERSION = 1;

let dbPromise: Promise<IDBPDatabase<WorkoutDB>> | null = null;

export const initDB = () => {
    if (!dbPromise) {
        dbPromise = openDB<WorkoutDB>(DB_NAME, DB_VERSION, {
            upgrade(db) {
                if (!db.objectStoreNames.contains('workouts')) {
                    const store = db.createObjectStore('workouts', { keyPath: 'id' });
                    store.createIndex('by-date', 'date');
                    store.createIndex('by-exercise', 'exerciseName');
                    // Add indexing on computed properties for quick heatmap/analytics lookups
                    store.createIndex('by-muscle', 'muscleGroup');
                }
                if (!db.objectStoreNames.contains('metadata')) {
                    db.createObjectStore('metadata');
                }
            },
        });
    }
    return dbPromise;
};

export const saveWorkouts = async (sets: WorkoutSet[]) => {
    const db = await initDB();
    const tx = db.transaction('workouts', 'readwrite');
    const store = tx.objectStore('workouts');

    // Clear old data to prevent duplication on re-imports, or we can handle merges later
    await store.clear();

    await Promise.all(sets.map((set) => store.put(set)));
    await tx.done;
};

export const getAllWorkouts = async (): Promise<WorkoutSet[]> => {
    const db = await initDB();
    return db.getAllFromIndex('workouts', 'by-date');
};

export const getWorkoutsByExercise = async (exerciseName: string): Promise<WorkoutSet[]> => {
    const db = await initDB();
    return db.getAllFromIndex('workouts', 'by-exercise', exerciseName);
};

export const hasStoredData = async (): Promise<boolean> => {
    const db = await initDB();
    const count = await db.count('workouts');
    return count > 0;
};
