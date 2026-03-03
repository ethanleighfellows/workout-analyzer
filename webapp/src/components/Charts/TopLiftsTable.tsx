import React, { useMemo } from 'react';
import { useStore } from '../../lib/store';

const calculateDOTS = (liftLbs: number, bwLbs: number, gender: 'M' | 'F'): string => {
    if (!bwLbs) return "0.00";
    const bwKg = bwLbs / 2.20462;
    const liftKg = liftLbs / 2.20462;

    const A = gender === 'M' ? -307.582 : -57.96288;
    const B = gender === 'M' ? 24.09 : 13.6175;
    const C = gender === 'M' ? -0.19278 : -0.112665;
    const D = gender === 'M' ? 0.0007391293 : 0.0005158568;
    const E = gender === 'M' ? -0.000001093 : -0.0000010706;

    const denom = A + B * bwKg + C * Math.pow(bwKg, 2) + D * Math.pow(bwKg, 3) + E * Math.pow(bwKg, 4);
    const coef = 500 / denom;
    return (liftKg * coef).toFixed(2);
};

export const TopLiftsTable: React.FC = () => {
    const workouts = useStore(state => state.filteredWorkouts);
    const athleteWeight = useStore(state => state.athleteWeight);
    const athleteGender = useStore(state => state.athleteGender);

    const topLifts = useMemo(() => {
        const maxWeights: Record<string, { weight: number, date: string }> = {};

        workouts.forEach(w => {
            if (!maxWeights[w.exerciseName] || w.weight > maxWeights[w.exerciseName].weight) {
                maxWeights[w.exerciseName] = { weight: w.weight, date: w.date.split(' ')[0] };
            }
        });

        return Object.entries(maxWeights)
            .sort((a, b) => b[1].weight - a[1].weight)
            .slice(0, 6); // Top 6 heaviest lifts
    }, [workouts]);

    if (!workouts.length) return null;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {topLifts.map(([exercise, data], i) => (
                <div key={exercise} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <div style={{ width: '24px', height: '24px', borderRadius: '50%', background: 'rgba(139, 92, 246, 0.2)', color: '#8b5cf6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', fontWeight: 700 }}>{i + 1}</div>
                        <span style={{ fontWeight: 500, fontSize: '0.9rem' }}>{exercise}</span>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                        <div style={{ fontWeight: 700 }}>{data.weight} lbs</div>
                        {athleteWeight ? (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', justifyContent: 'flex-end', fontSize: '0.75rem' }}>
                                <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>{calculateDOTS(data.weight, athleteWeight, athleteGender)} DOTS</span>
                                <span style={{ color: 'var(--text-secondary)' }}>• {data.date}</span>
                            </div>
                        ) : (
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{data.date}</div>
                        )}
                    </div>
                </div>
            ))}
        </div>
    );
};
