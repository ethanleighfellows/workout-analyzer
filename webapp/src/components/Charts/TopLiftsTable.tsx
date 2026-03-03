import React, { useMemo } from 'react';
import { useStore } from '../../lib/store';

export const TopLiftsTable: React.FC = () => {
    const workouts = useStore(state => state.filteredWorkouts);

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
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{data.date}</div>
                    </div>
                </div>
            ))}
        </div>
    )
}
