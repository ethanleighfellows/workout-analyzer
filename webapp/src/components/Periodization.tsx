import React, { useMemo } from 'react';
import {
    Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip as ChartTooltip, Legend, Filler, BarElement
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { useStore } from '../lib/store';
import { Activity, ShieldAlert, Zap, TrendingUp } from 'lucide-react';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, ChartTooltip, Legend, Filler);

export const Periodization: React.FC = () => {
    const summary = useStore(state => state.filteredSummary);
    if (!summary || !summary.acwrData || !summary.mesocycles || !summary.recoveryStatus || !summary.densityData) return null;

    // --- ACWR Chart ---
    const acwrChartData = useMemo(() => {
        // filter 0 ratios to prevent noise
        const filtered = summary.acwrData!.filter(d => d.ratio > 0);
        return {
            labels: filtered.map(d => d.date),
            datasets: [
                {
                    label: 'Acute:Chronic Workload Ratio (ACWR)',
                    data: filtered.map(d => d.ratio),
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHitRadius: 10
                }
            ]
        };
    }, [summary.acwrData]);

    const acwrOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: { mode: 'index' as const, intersect: false },
            annotation: { // Conceptual line at 1.5 danger zone
                annotations: {
                    line1: { type: 'line', yMin: 1.5, yMax: 1.5, borderColor: '#ef4444', borderWidth: 2, borderDash: [5, 5] },
                    line2: { type: 'line', yMin: 0.8, yMax: 0.8, borderColor: '#f59e0b', borderWidth: 2, borderDash: [5, 5] }
                }
            }
        },
        scales: {
            x: { grid: { display: false }, ticks: { display: false } },
            y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, min: 0 }
        }
    };

    // --- Density Chart ---
    const densityChartData = useMemo(() => {
        return {
            labels: summary.densityData!.map(d => d.date),
            datasets: [
                {
                    label: 'Workout Density (lbs/min)',
                    data: summary.densityData!.map(d => d.density),
                    backgroundColor: '#8b5cf6',
                    borderRadius: 4
                }
            ]
        };
    }, [summary.densityData]);

    const densityOptions = {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { display: false }, ticks: { display: false } },
            y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }
        }
    };

    // --- Current Status Blocks ---
    const latestACWR = summary.acwrData![summary.acwrData!.length - 1]?.ratio || 0;
    let acwrStatus = { text: 'Optimal', color: '#10b981', icon: <Activity size={20} /> };
    if (latestACWR > 1.5) acwrStatus = { text: 'Overtraining Risk', color: '#ef4444', icon: <ShieldAlert size={20} /> };
    if (latestACWR < 0.8) acwrStatus = { text: 'Undertraining', color: '#f59e0b', icon: <TrendingUp size={20} /> };

    const sortedRecovery = Object.entries(summary.recoveryStatus!).sort((a, b) => a[1].recoveryPct - b[1].recoveryPct);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {/* Top Insight Row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr) minmax(300px, 1fr)', gap: '20px' }}>
                <div className="glass-panel" style={{ padding: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <h3 style={{ fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Current ACWR Status</h3>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: acwrStatus.color }}>
                            {acwrStatus.icon}
                            <span style={{ fontWeight: 600 }}>{acwrStatus.text} ({latestACWR.toFixed(2)})</span>
                        </div>
                    </div>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        Acute-to-Chronic Workload Ratio compares your last 7 days of training volume against your 28-day historical rolling average. Ratios between 0.8 and 1.3 are the 'sweet spot'. Above 1.5 indicates a drastic spike in workload and heavily increases injury risk.
                    </p>
                </div>

                <div className="glass-panel" style={{ padding: '24px' }}>
                    <h3 style={{ fontSize: '1.2rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>Mesocycles Detected</h3>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px' }}>
                        <span style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--accent-primary)' }}>{summary.mesocycles!.length}</span>
                        <span style={{ color: 'var(--text-secondary)' }}>Training Blocks</span>
                    </div>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '8px' }}>
                        Automatically segmented based on rest gaps &gt;= 7 days. Your longest block consisted of {Math.max(...summary.mesocycles!.map(m => m.count))} workouts.
                    </p>
                </div>
            </div>

            {/* Charts Row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>
                <div className="glass-panel" style={{ padding: '24px', minHeight: '350px' }}>
                    <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Athletic Workload Trajectory (ACWR)</h3>
                    <div style={{ height: '300px' }}>
                        <Line data={acwrChartData} options={acwrOptions} />
                    </div>
                </div>

                <div className="glass-panel" style={{ padding: '24px', minHeight: '350px' }}>
                    <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Systemic Density (lbs/min)</h3>
                    <div style={{ height: '300px' }}>
                        <Bar data={densityChartData} options={densityOptions} />
                    </div>
                </div>
            </div>

            {/* Recovery Heatmap */}
            <div className="glass-panel" style={{ padding: '24px' }}>
                <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Zap size={20} color="#eab308" /> Muscle Readiness Heatmap
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '12px' }}>
                    {sortedRecovery.map(([muscle, data]) => {
                        let color = '#ef4444'; // red
                        if (data.recoveryPct > 50) color = '#f59e0b'; // orange
                        if (data.recoveryPct > 80) color = '#10b981'; // green

                        return (
                            <div key={muscle} style={{ padding: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                                <div style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '8px' }}>{muscle}</div>
                                <div style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.1)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                                    <div style={{ width: `${data.recoveryPct}%`, height: '100%', backgroundColor: color, transition: 'width 1s ease' }} />
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '8px', textAlign: 'right' }}>
                                    {data.recoveryPct}% Recovered
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>

        </div>
    );
};
