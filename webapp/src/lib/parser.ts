import Papa from 'papaparse';
import { WorkoutSet } from '../types/workout';
import { ExerciseClassifier } from './classifier';
import { calculateE1RM_Brzycki, calculateE1RM_Epley, calculateIntensityPercentage } from './analytics';

export const parseStrongCSV = (file: File): Promise<WorkoutSet[]> => {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            complete: (results) => {
                try {
                    const parsedSets: WorkoutSet[] = results.data.map((row: any, index: number) => {
                        const exerciseName = row['Exercise Name'] || row['Exercise'] || 'Unknown';
                        const classification = ExerciseClassifier.classify(exerciseName);
                        const weight = parseFloat(row['Weight']) || 0;
                        const reps = parseInt(row['Reps'], 10) || 0;

                        const rpeStr = row['RPE'];
                        const rpe = rpeStr ? parseFloat(rpeStr) : undefined;

                        return {
                            id: `${row['Date']}-${exerciseName}-${index}`, // Guarantee unique ID
                            date: row['Date'], // Maintain raw ISO or Date string from Strong
                            workoutName: row['Workout Name'] || 'Workout',
                            durationMinutes: row['Duration'] ? parseFloat(row['Duration'].replace('m', '')) : 0,
                            exerciseName,
                            setOrder: parseInt(row['Set Order'], 10) || 1,
                            weight,
                            reps,
                            distance: row['Distance'] ? parseFloat(row['Distance']) : undefined,
                            seconds: row['Seconds'] ? parseInt(row['Seconds'], 10) : undefined,
                            notes: row['Notes'] || '',
                            workoutNotes: row['Workout Notes'] || '',
                            rpe,
                            e1rmEpley: weight && reps ? calculateE1RM_Epley(weight, reps) : undefined,
                            e1rmBrzycki: weight && reps ? calculateE1RM_Brzycki(weight, reps) : undefined,
                            intensityPct: reps ? calculateIntensityPercentage(reps) : undefined,
                            muscleGroup: classification.muscleGroup,
                            exerciseCategory: classification.category,
                            isUnilateral: classification.isUnilateral
                        };
                    });

                    resolve(parsedSets);
                } catch (error) {
                    reject(new Error('Failed to parse Strong format CSV. Invalid columns.'));
                }
            },
            error: (error) => {
                reject(error);
            }
        });
    });
};
