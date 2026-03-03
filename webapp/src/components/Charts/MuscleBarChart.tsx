import React, { useMemo } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { useStore } from '../../lib/store';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

export const MuscleBarChart: React.FC = () => {
    const workouts = useStore(state => state.filteredWorkouts);

    const chartData = useMemo(() => {
        // Aggregate volume by muscle group
        const volumeByMuscle = workouts.reduce((acc, curr) => {
            const muscle = curr.muscleGroup || 'Other';
            if (!acc[muscle]) acc[muscle] = 0;
            acc[muscle] += (curr.weight * curr.reps);
            return acc;
        }, {} as Record<string, number>);

        // Sort descending
        const sortedMuscles = Object.entries(volumeByMuscle).sort((a, b) => b[1] - a[1]);
        const labels = sortedMuscles.map(m => m[0]);
        const data = sortedMuscles.map(m => m[1]);

        return {
            labels,
            datasets: [
                {
                    label: 'Total Volume',
                    data,
                    backgroundColor: 'rgba(139, 92, 246, 0.8)',
                    borderRadius: 4,
                    hoverBackgroundColor: 'rgba(139, 92, 246, 1)'
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
            x: {
                grid: { display: false },
                ticks: { color: '#a1a1aa' }
            },
            y: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: {
                    color: '#a1a1aa',
                    callback: function (value: any) {
                        return (value / 1000).toFixed(1) + 'k';
                    }
                },
                beginAtZero: true
            }
        }
    };

    if (!workouts.length) return <div>No data to display.</div>;

    return <Bar data={chartData} options={options} />;
};
