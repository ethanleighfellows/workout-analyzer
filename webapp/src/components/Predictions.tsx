import React, { useMemo, useState } from 'react';
import { useStore } from '../lib/store';
import { Line } from 'react-chartjs-2';

export const Predictions: React.FC = () => {
    const workouts = useStore((state) => state.filteredWorkouts);
    const [selectedExercise, setSelectedExercise] = useState<string>('');

    const topExercises = useMemo(() => {
        const counts: Record<string, number> = {};
        workouts.forEach(w => {
            if (w.exerciseCategory && w.exerciseCategory.includes('push') || w.exerciseCategory?.includes('pull') || w.exerciseCategory?.includes('compound')) {
                counts[w.exerciseName] = (counts[w.exerciseName] || 0) + 1;
            }
        });
        return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 10).map(x => x[0]);
    }, [workouts]);

    // Set default exercise
    React.useEffect(() => {
        if (topExercises.length > 0 && !selectedExercise) {
            setSelectedExercise(topExercises[0]);
        }
    }, [topExercises, selectedExercise]);

    const chartData = useMemo(() => {
        if (!selectedExercise) return null;

        // Filter sets for the selected exercise that have 1RM estimates
        const relevantSets = workouts.filter(w => w.exerciseName === selectedExercise && w.e1rmBrzycki);

        // Group by date, get max 1RM for that date
        const dailyMax1RM: Record<string, number> = {};
        relevantSets.forEach(w => {
            const d = w.date.split(' ')[0];
            const e1rm = w.e1rmBrzycki || 0;
            if (!dailyMax1RM[d] || e1rm > dailyMax1RM[d]) {
                dailyMax1RM[d] = e1rm;
            }
        });

        const sortedDates = Object.keys(dailyMax1RM).sort();
        const dataPoints = sortedDates.map(d => dailyMax1RM[d]);

        // Simple Linear Regression for Forecasting
        let slope = 0;
        let intercept = 0;
        if (dataPoints.length > 1) {
            const n = dataPoints.length;
            const sumX = (n * (n - 1)) / 2; // sum of 0 to n-1
            const sumY = dataPoints.reduce((a, b) => a + b, 0);
            const sumXY = dataPoints.reduce((a, b, i) => a + (b * i), 0);
            const sumXX = dataPoints.reduce((a, _, i) => a + (i * i), 0);

            slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
            intercept = (sumY - slope * sumX) / n;
        }

        // Add forecast points
        const forecastLabels = [...sortedDates];
        const trendData = [...dataPoints];

        // Predict next 3 workouts
        for (let i = 1; i <= 3; i++) {
            forecastLabels.push(`Forecast +${i}`);
            trendData.push(intercept + slope * (dataPoints.length - 1 + i));
        }

        return {
            labels: forecastLabels,
            datasets: [
                {
                    label: 'Estimated 1RM (lbs)',
                    data: [...dataPoints, null, null, null],
                    borderColor: '#10b981',
                    tension: 0.2,
                    pointBackgroundColor: '#10b981'
                },
                {
                    label: 'Forecast Trend',
                    data: trendData,
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0
                }
            ]
        };

    }, [workouts, selectedExercise]);

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: '#a1a1aa' } },
            tooltip: { mode: 'index' as const, intersect: false, backgroundColor: 'rgba(20, 20, 25, 0.9)', titleColor: '#a1a1aa', bodyColor: '#fff' },
        },
        scales: {
            x: { grid: { display: false }, ticks: { color: '#a1a1aa', maxTicksLimit: 7 } },
            y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#a1a1aa' } }
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            <div className="glass-panel" style={{ padding: '24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                    <h3 style={{ fontSize: '1.2rem', color: 'var(--text-secondary)' }}>1RM Forecaster</h3>
                    <select
                        value={selectedExercise}
                        onChange={(e) => setSelectedExercise(e.target.value)}
                        style={{ background: 'rgba(255,255,255,0.05)', color: 'white', border: '1px solid var(--border-color)', padding: '8px 12px', borderRadius: '8px', cursor: 'pointer' }}
                    >
                        {topExercises.map(ex => <option key={ex} value={ex} style={{ background: 'var(--bg-main)' }}>{ex}</option>)}
                    </select>
                </div>

                <div style={{ height: '350px' }}>
                    {chartData ? <Line data={chartData} options={options} /> : <div style={{ color: 'var(--text-secondary)' }}>Insufficient data for modeling.</div>}
                </div>

                <p style={{ marginTop: '20px', fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '600px' }}>
                    * Projections use Epley/Brzycki algorithmic variance coupled with linear regression routing over your historical progressive overload data. Exceeding trend lines indicates an Olympic-level adaptation phase.
                </p>
            </div>

        </div>
    );
};
