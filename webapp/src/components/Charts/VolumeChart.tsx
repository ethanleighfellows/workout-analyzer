import React, { useMemo } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler,
    Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { useStore } from '../../lib/store';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler,
    Legend
);

export const VolumeChart: React.FC = () => {
    const workouts = useStore(state => state.filteredWorkouts);

    const chartData = useMemo(() => {
        // Group workouts by date and sum the volume
        const volumeByDate = workouts.reduce((acc, curr) => {
            const dateStr = curr.date.split(' ')[0]; // YYYY-MM-DD
            if (!acc[dateStr]) acc[dateStr] = 0;
            acc[dateStr] += (curr.weight * curr.reps);
            return acc;
        }, {} as Record<string, number>);

        // Sort dates
        const sortedDates = Object.keys(volumeByDate).sort();

        // Smooth the line (optional: rolling average could go here)
        const dataPoints = sortedDates.map(date => volumeByDate[date]);

        return {
            labels: sortedDates,
            datasets: [
                {
                    fill: true,
                    label: 'Daily Volume (lbs)',
                    data: dataPoints,
                    borderColor: 'rgba(79, 70, 229, 1)',
                    backgroundColor: 'rgba(79, 70, 229, 0.2)',
                    tension: 0.4, // Smooth curve
                    pointRadius: 3,
                    pointBackgroundColor: 'rgba(139, 92, 246, 1)'
                }
            ]
        };
    }, [workouts]);

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
            tooltip: {
                mode: 'index' as const,
                intersect: false,
                backgroundColor: 'rgba(20, 20, 25, 0.9)',
                titleColor: '#a1a1aa',
                bodyColor: '#fff',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                padding: 12,
                displayColors: false,
            },
        },
        scales: {
            x: {
                grid: {
                    display: false,
                },
                ticks: {
                    color: '#a1a1aa',
                    maxTicksLimit: 7
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                },
                ticks: {
                    color: '#a1a1aa',
                    callback: function (value: any) {
                        return (value / 1000).toFixed(1) + 'k';
                    }
                },
                beginAtZero: true
            }
        },
        interaction: {
            mode: 'nearest' as const,
            axis: 'x' as const,
            intersect: false
        }
    };

    if (!workouts.length) return <div>No data to display.</div>;

    return <Line options={options} data={chartData} />;
};
