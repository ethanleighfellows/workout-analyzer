import React, { useMemo } from 'react';
import {
    Chart as ChartJS,
    RadialLinearScale,
    PointElement,
    LineElement,
    Filler,
    Tooltip,
    Legend,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';
import { useStore } from '../../lib/store';

ChartJS.register(
    RadialLinearScale,
    PointElement,
    LineElement,
    Filler,
    Tooltip,
    Legend
);

export const MuscleRadarChart: React.FC = () => {
    const workouts = useStore(state => state.filteredWorkouts);

    const chartData = useMemo(() => {
        // Aggregate volume by muscle group
        const volumeByMuscle = workouts.reduce((acc, curr) => {
            const muscle = curr.muscleGroup || 'Other';
            if (!acc[muscle]) acc[muscle] = 0;
            acc[muscle] += (curr.weight * curr.reps);
            return acc;
        }, {} as Record<string, number>);

        const labels = Object.keys(volumeByMuscle).filter(m => m !== 'Other');
        const data = labels.map(l => volumeByMuscle[l]);

        return {
            labels,
            datasets: [
                {
                    label: 'Total Volume',
                    data,
                    backgroundColor: 'rgba(139, 92, 246, 0.2)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    pointBackgroundColor: 'rgba(79, 70, 229, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(79, 70, 229, 1)',
                    borderWidth: 2,
                },
            ],
        };
    }, [workouts]);

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(20, 20, 25, 0.9)',
                bodyColor: '#fff',
                titleColor: '#a1a1aa',
                padding: 12,
                callbacks: {
                    label: (context: any) => `${(context.raw / 1000).toFixed(1)}k lbs`
                }
            }
        },
        scales: {
            r: {
                angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                pointLabels: {
                    color: '#a1a1aa',
                    font: { size: 12 }
                },
                ticks: {
                    display: false, // hide the scale numbers on the radar lines
                }
            }
        }
    };

    if (!workouts.length) return <div>No data to display.</div>;

    return <Radar data={chartData} options={options} />;
};
