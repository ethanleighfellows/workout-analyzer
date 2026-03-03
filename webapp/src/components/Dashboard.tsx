import React from 'react';
import { useStore, DateFilter } from '../lib/store';
import { Activity, Dumbbell, Calendar, Target, Filter } from 'lucide-react';
import { VolumeChart } from './Charts/VolumeChart';
import { MuscleBarChart } from './Charts/MuscleBarChart';
import { ActivityFreqChart } from './Charts/ActivityFreqChart';
import { TopLiftsTable } from './Charts/TopLiftsTable';

export const Dashboard: React.FC = () => {
    const summary = useStore((state) => state.filteredSummary);
    const dateFilter = useStore((state) => state.dateFilter);
    const setDateFilter = useStore((state) => state.setDateFilter);

    if (!summary) return null;

    const filters: { value: DateFilter, label: string }[] = [
        { value: 'all', label: 'All Time' },
        { value: '1y', label: 'Last Year' },
        { value: '6m', label: 'Last 6 Months' },
        { value: '3m', label: 'Last 3 Months' },
        { value: '1m', label: 'Last Month' }
    ];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {/* Slicer & KPI Row */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(255,255,255,0.05)', padding: '6px', borderRadius: '12px' }}>
                    <Filter size={18} color="var(--text-secondary)" style={{ marginLeft: '8px' }} />
                    {filters.map(f => (
                        <button
                            key={f.value}
                            onClick={() => setDateFilter(f.value)}
                            style={{
                                background: dateFilter === f.value ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                                color: dateFilter === f.value ? '#8b5cf6' : 'var(--text-secondary)',
                                border: 'none',
                                padding: '6px 16px',
                                borderRadius: '8px',
                                cursor: 'pointer',
                                fontWeight: dateFilter === f.value ? 600 : 500,
                                transition: 'all 0.2s',
                                fontSize: '0.85rem'
                            }}
                        >
                            {f.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* KPI Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px' }}>
                <StatCard
                    icon={<Calendar color="#8b5cf6" size={24} />}
                    label="Total Workouts"
                    value={summary.totalWorkouts.toLocaleString()}
                />
                <StatCard
                    icon={<Dumbbell color="#4f46e5" size={24} />}
                    label="Total Volume"
                    value={`${(summary.totalVolume / 1000).toFixed(1)}k lbs`}
                />
                <StatCard
                    icon={<Activity color="#ef4444" size={24} />}
                    label="Total Sets"
                    value={summary.totalSets.toLocaleString()}
                />
                <StatCard
                    icon={<Target color="#10b981" size={24} />}
                    label="Unique Exercises"
                    value={summary.uniqueExercises.toLocaleString()}
                />
            </div>

            {/* Charts Grid Row 1 */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px', marginTop: '10px' }}>
                <div className="glass-panel" style={{ padding: '24px', minHeight: '350px' }}>
                    <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Volume Over Time</h3>
                    <div style={{ height: '300px' }}>
                        <VolumeChart />
                    </div>
                </div>

                <div className="glass-panel" style={{ padding: '24px', minHeight: '350px' }}>
                    <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Muscle Volume Distribution</h3>
                    <div style={{ height: '300px' }}>
                        <MuscleBarChart />
                    </div>
                </div>
            </div>

            {/* Charts Grid Row 2 */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', marginTop: '10px' }}>
                <div className="glass-panel" style={{ padding: '24px', minHeight: '350px' }}>
                    <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Workout Frequency (Monthly)</h3>
                    <div style={{ height: '300px' }}>
                        <ActivityFreqChart />
                    </div>
                </div>

                <div className="glass-panel" style={{ padding: '24px', minHeight: '350px' }}>
                    <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Heaviest Lifts (All Time)</h3>
                    <div style={{ height: '300px', overflowY: 'auto', paddingRight: '8px' }}>
                        <TopLiftsTable />
                    </div>
                </div>
            </div>

        </div>
    );
};

const StatCard: React.FC<{ icon: React.ReactNode, label: string, value: string | number }> = ({ icon, label, value }) => (
    <div className="glass-panel" style={{ padding: '24px', display: 'flex', alignItems: 'center', gap: '20px' }}>
        <div style={{
            backgroundColor: 'rgba(255,255,255,0.05)',
            padding: '16px',
            borderRadius: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
        }}>
            {icon}
        </div>
        <div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '4px' }}>{label}</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 'bold' }}>{value}</div>
        </div>
    </div>
);
