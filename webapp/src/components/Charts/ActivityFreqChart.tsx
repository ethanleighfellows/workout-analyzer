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

export const ActivityFreqChart: React.FC = () => {
    const workouts = useStore(state => state.filteredWorkouts);

    const chartData = useMemo(() => {
        // Aggregate unique workouts by Month-Year
        const uniqueWorkouts = new Set<string>();
        const monthCounts: Record<string, number> = {};

        workouts.forEach(w => {
            // e.g. "2024-05-20 16:48:07"
            const datePart = w.date.split(' ')[0];
            const monthYear = datePart.substring(0, 7); // "YYYY-MM"

            const uniqueKey = `${datePart}-${w.workoutName}`;
            if (!uniqueWorkouts.has(uniqueKey)) {
                uniqueWorkouts.add(uniqueKey);
                if (!monthCounts[monthYear]) monthCounts[monthYear] = 0;
                monthCounts[monthYear]++;
            }
        });

        const sortedMonths = Object.keys(monthCounts).sort();
        const data = sortedMonths.map(m => monthCounts[m]);

        return {
            labels: sortedMonths,
            datasets: [
                {
                    label: 'Workouts',
                    data,
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    borderRadius: 4,
                    hoverBackgroundColor: 'rgba(16, 185, 129, 1)'
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
                    stepSize: 1
                },
                beginAtZero: true
            }
        }
    };

    if (!workouts.length) return <div>No data to display.</div>;

    return <Bar data={chartData} options={options} />;
};
