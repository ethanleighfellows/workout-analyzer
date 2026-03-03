import React, { useMemo } from 'react';
import { useStore } from '../lib/store';
import { AlertTriangle, CheckCircle, Award } from 'lucide-react';
import {
    Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler, Tooltip as ChartTooltip, Legend
} from 'chart.js';
import { Radar } from 'react-chartjs-2';

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, ChartTooltip, Legend);

export const Insights: React.FC = () => {
    const summary = useStore(state => state.filteredSummary);
    const workouts = useStore(state => state.filteredWorkouts);

    if (!summary || !summary.plateaus || !summary.archetype) return null;

    // --- Plateau Analysis ---
    const stalledExercises = summary.plateaus.filter(p => p.status === 'stalled');
    const progressingExercises = summary.plateaus.filter(p => p.status === 'progressing');

    // --- Elite Archetype Radar ---
    const radarData = {
        labels: ['Lower Body (Squat/Hinge)', 'Horizontal/Vertical Push', 'Horizontal/Vertical Pull', 'Isolation/Accessory'],
        datasets: [
            {
                label: 'Volume Distribution Ratio',
                data: [
                    summary.archetype.lower * 100,
                    summary.archetype.push * 100,
                    summary.archetype.pull * 100,
                    summary.archetype.isolation * 100
                ],
                backgroundColor: 'rgba(139, 92, 246, 0.2)',
                borderColor: '#8b5cf6',
                borderWidth: 2,
                pointBackgroundColor: '#8b5cf6',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#8b5cf6',
                fill: true,
            },
        ],
    };

    const radarOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            r: {
                angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                pointLabels: { color: '#a1a1aa', font: { size: 12 } },
                ticks: { display: false, min: 0, max: 100 }
            }
        },
        plugins: { legend: { display: false } }
    };

    // --- PR Timeline Generator ---
    const prTimeline = useMemo(() => {
        const history: { date: string, exercise: string, weight: number }[] = [];
        const maxes: Record<string, number> = {};

        // Sort workouts oldest to newest
        const sorted = [...workouts].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

        sorted.forEach(w => {
            if (!maxes[w.exerciseName] || w.weight > maxes[w.exerciseName]) {
                // If it's a new PR (and not just the first time doing it at a tiny weight)
                if (maxes[w.exerciseName] && w.weight > maxes[w.exerciseName]) {
                    history.push({ date: w.date.split(' ')[0], exercise: w.exerciseName, weight: w.weight });
                }
                maxes[w.exerciseName] = w.weight;
            }
        });

        // Return latest 10 PRs descending
        return history.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()).slice(0, 10);
    }, [workouts]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {/* Top Analysis Row */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>

                {/* Archetype Radar */}
                <div className="glass-panel" style={{ padding: '24px' }}>
                    <h3 style={{ marginBottom: '8px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Athletic Archetype</h3>
                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-primary)', marginBottom: '20px' }}>
                        {summary.archetype.archetype}
                    </div>
                    <div style={{ height: '300px' }}>
                        <Radar data={radarData} options={radarOptions} />
                    </div>
                </div>

                {/* Sticking Point Detection */}
                <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>1RM Sticking Point Analysis</h3>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '20px' }}>
                        The algorithm monitors the Est 1RM of large compound movements over the last 4 weeks. If progression is ≤ 2%, the movement is flagged as a plateau requiring a deload or variation switch.
                    </p>

                    <div style={{ overflowY: 'auto', flexGrow: 1, paddingRight: '8px' }}>
                        <div style={{ marginBottom: '16px' }}>
                            <h4 style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#ef4444', marginBottom: '12px' }}>
                                <AlertTriangle size={18} /> Stalled Movements ({stalledExercises.length})
                            </h4>
                            {stalledExercises.map(ex => (
                                <div key={ex.exercise} style={{ padding: '10px 14px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '6px', marginBottom: '8px', fontSize: '0.9rem' }}>
                                    {ex.exercise}
                                </div>
                            ))}
                            {stalledExercises.length === 0 && <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>No plateaus detected!</div>}
                        </div>

                        <div>
                            <h4 style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981', marginBottom: '12px' }}>
                                <CheckCircle size={18} /> Progressing Movements ({progressingExercises.length})
                            </h4>
                            {progressingExercises.slice(0, 5).map(ex => (
                                <div key={ex.exercise} style={{ padding: '10px 14px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '6px', marginBottom: '8px', fontSize: '0.9rem' }}>
                                    {ex.exercise}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* PR Timeline Feed */}
            <div className="glass-panel" style={{ padding: '24px' }}>
                <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Award size={20} color="#f59e0b" /> Lifetime PR Timeline
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {prTimeline.map((pr, i) => (
                        <div key={`${pr.date}-${pr.exercise}-${i}`} style={{ display: 'flex', alignItems: 'center', gap: '16px', padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', borderLeft: '4px solid #f59e0b' }}>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', minWidth: '100px' }}>{pr.date}</div>
                            <div style={{ flexGrow: 1, fontWeight: 500 }}>{pr.exercise}</div>
                            <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)' }}>{pr.weight} lbs</div>
                        </div>
                    ))}
                    {prTimeline.length === 0 && <div style={{ color: 'var(--text-secondary)' }}>Not enough historical data to generate PR feed.</div>}
                </div>
            </div>
        </div>
    );
};
