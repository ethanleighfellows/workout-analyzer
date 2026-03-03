import React, { useMemo } from 'react';
import { useStore } from '../lib/store';

export const Biomechanics: React.FC = () => {
    const workouts = useStore((state) => state.filteredWorkouts);

    const stats = useMemo(() => {
        let unilateralVolume = 0;
        let bilateralVolume = 0;
        const categoryVolume: Record<string, number> = {};
        const muscleVolume: Record<string, number> = {};

        workouts.forEach(w => {
            const vol = w.weight * w.reps;

            // Unilateral vs Bilateral
            if (w.isUnilateral) unilateralVolume += vol;
            else bilateralVolume += vol;

            // Category
            const cat = w.exerciseCategory || 'isolation';
            if (!categoryVolume[cat]) categoryVolume[cat] = 0;
            categoryVolume[cat] += vol;

            // Muscle group breakdown
            const musc = w.muscleGroup || 'Other';
            if (!muscleVolume[musc]) muscleVolume[musc] = 0;
            muscleVolume[musc] += vol;
        });

        const totalVolume = unilateralVolume + bilateralVolume || 1; // Prevent divide by 0

        // Calculate Push to Pull ratio
        const horizPush = categoryVolume['horizontal_push'] || 0;
        const vertPush = categoryVolume['vertical_push'] || 0;
        const horizPull = categoryVolume['horizontal_pull'] || 0;
        const vertPull = categoryVolume['vertical_pull'] || 0;

        const totalPush = horizPush + vertPush;
        const totalPull = horizPull + vertPull;
        const pushPullRatio = totalPull > 0 ? (totalPush / totalPull).toFixed(2) : 'N/A';

        return {
            unilateralPct: Math.round((unilateralVolume / totalVolume) * 100),
            bilateralPct: Math.round((bilateralVolume / totalVolume) * 100),
            pushPullRatio,
            totalPush,
            totalPull,
            muscleVolume
        };
    }, [workouts]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>

                {/* Symmetry Analysis */}
                <div className="glass-panel" style={{ padding: '24px' }}>
                    <h3 style={{ marginBottom: '16px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Symmetry Analysis</h3>

                    <div style={{ marginBottom: '24px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                            <span>Unilateral Movements (Indiv. Limbs)</span>
                            <span style={{ fontWeight: 600 }}>{stats.unilateralPct}%</span>
                        </div>
                        <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ width: `${stats.unilateralPct}%`, height: '100%', background: 'var(--accent-primary)' }} />
                        </div>
                    </div>

                    <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                            <span>Bilateral Movements (Both Limbs)</span>
                            <span style={{ fontWeight: 600 }}>{stats.bilateralPct}%</span>
                        </div>
                        <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ width: `${stats.bilateralPct}%`, height: '100%', background: 'var(--accent-secondary)' }} />
                        </div>
                    </div>

                    <p style={{ marginTop: '20px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        {stats.unilateralPct < 20 ? 'Recommendation: Incorporate more unilateral exercises (e.g. Bulgarian Split Squats, Single Arm DB Rows) to prevent muscular imbalances.' : 'Great job maintaining unilateral balance.'}
                    </p>
                </div>

                {/* Push vs Pull Balance */}
                <div className="glass-panel" style={{ padding: '24px' }}>
                    <h3 style={{ marginBottom: '16px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Push/Pull Balance</h3>

                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '20px 0' }}>
                        <div style={{ fontSize: '2.5rem', fontWeight: 'bold' }}>{stats.pushPullRatio}</div>
                        <div style={{ marginLeft: '10px', fontSize: '1rem', color: 'var(--text-secondary)' }}>Ratio</div>
                    </div>

                    <div style={{ display: 'flex', gap: '2px', height: '24px', borderRadius: '12px', overflow: 'hidden', marginBottom: '12px' }}>
                        <div style={{ width: `${(stats.totalPush / ((stats.totalPush + stats.totalPull) || 1)) * 100}%`, background: '#ef4444' }} />
                        <div style={{ width: `${(stats.totalPull / ((stats.totalPush + stats.totalPull) || 1)) * 100}%`, background: '#10b981' }} />
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                        <span style={{ color: '#ef4444' }}>Push ({(stats.totalPush / 1000).toFixed(1)}k)</span>
                        <span style={{ color: '#10b981' }}>Pull ({(stats.totalPull / 1000).toFixed(1)}k)</span>
                    </div>

                </div>

            </div>

            {/* Muscle Heatmap Breakdown */}
            <div className="glass-panel" style={{ padding: '24px' }}>
                <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Muscle Activation Load (Volume Base)</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px' }}>
                    {Object.entries(stats.muscleVolume).sort((a, b) => b[1] - a[1]).map(([muscle, vol]) => (
                        <div key={muscle} style={{ background: 'rgba(255,255,255,0.05)', padding: '16px', borderRadius: '12px' }}>
                            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>{muscle}</div>
                            <div style={{ fontSize: '1.3rem', fontWeight: 600 }}>{(vol / 1000).toFixed(1)}k lbs</div>
                        </div>
                    ))}
                </div>
            </div>

        </div>
    );
};
