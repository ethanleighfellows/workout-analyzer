import React, { useEffect, useState } from 'react';
import { useStore } from './lib/store';
import { FileUpload } from './components/FileUpload';
import { Dashboard } from './components/Dashboard';
import { Biomechanics } from './components/Biomechanics';
import { Predictions } from './components/Predictions';
import { Periodization } from './components/Periodization';
import { Insights } from './components/Insights';
import { Activity, Compass, Flame, History, Settings, LogOut, BarChart2, TrendingUp } from 'lucide-react';

function App() {
    const { initialize, isLoaded, workouts, clearData } = useStore();
    const [activeTab, setActiveTab] = useState('dashboard');

    useEffect(() => {
        initialize();
    }, [initialize]);

    const navItems = [
        { id: 'dashboard', label: 'Dashboard', icon: <Compass size={20} /> },
        { id: 'biomechanics', label: 'Biomechanics', icon: <Activity size={20} /> },
        { id: 'periodization', label: 'Periodization', icon: <BarChart2 size={20} /> },
        { id: 'insights', label: 'Insights & Archetypes', icon: <TrendingUp size={20} /> },
        { id: 'predictions', label: 'Predictions (1RM)', icon: <Flame size={20} /> },
        { id: 'history', label: 'History log', icon: <History size={20} /> },
    ];

    return (
        <div className="app-container" style={{ display: 'flex', minHeight: '100vh' }}>
            {/* Sidebar Navigation */}
            <nav className="glass-panel" style={{
                width: '260px',
                margin: '20px',
                padding: '24px',
                display: 'flex',
                flexDirection: 'column',
                gap: '30px',
                position: 'sticky',
                top: '20px',
                height: 'calc(100vh - 40px)',
                zIndex: 10
            }}>
                <div>
                    <h1 style={{ fontSize: '1.4rem', fontWeight: 800, background: 'linear-gradient(135deg, #4f46e5, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.5px', lineHeight: '1.2' }}>
                        ETHAN'S WORKOUT ANALYTICS
                    </h1>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1 }}>
                    <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-secondary)', letterSpacing: '1px', marginBottom: '8px', paddingLeft: '12px' }}>Menu</div>
                    {navItems.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => setActiveTab(item.id)}
                            style={{
                                background: activeTab === item.id ? 'rgba(79, 70, 229, 0.15)' : 'transparent',
                                color: activeTab === item.id ? 'white' : 'var(--text-secondary)',
                                border: 'none',
                                padding: '12px 16px',
                                borderRadius: '12px',
                                textAlign: 'left',
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                                fontWeight: activeTab === item.id ? 600 : 500,
                                display: 'flex',
                                alignItems: 'center',
                                gap: '12px',
                                boxShadow: activeTab === item.id ? 'inset 2px 0 0 0 #8b5cf6' : 'none'
                            }}
                        >
                            {React.cloneElement(item.icon, { color: activeTab === item.id ? '#8b5cf6' : 'currentColor' })}
                            {item.label}
                        </button>
                    ))}
                </div>

                {/* Bottom Actions */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <button style={{ background: 'transparent', color: 'var(--text-secondary)', border: 'none', padding: '12px 16px', display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer', borderRadius: '12px', transition: 'all 0.2s', textAlign: 'left' }} className="hover-highlight">
                        <Settings size={20} /> Settings
                    </button>
                    {isLoaded && workouts.length > 0 && (
                        <button onClick={clearData} style={{ background: 'transparent', color: '#ef4444', border: 'none', padding: '12px 16px', display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer', borderRadius: '12px', transition: 'all 0.2s', textAlign: 'left' }} className="hover-highlight">
                            <LogOut size={20} /> Clear Data
                        </button>
                    )}
                </div>
            </nav>

            {/* Main Content Area */}
            <main style={{ flex: 1, padding: '20px 20px 20px 0', maxWidth: 'calc(100vw - 300px)' }}>
                <div className="glass-panel" style={{ minHeight: '100%', padding: '32px' }}>

                    <header style={{ marginBottom: '40px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                            <h2 style={{ fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.5px' }}>
                                {navItems.find(i => i.id === activeTab)?.label || 'Dashboard'}
                            </h2>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>
                                {!isLoaded || workouts.length === 0 ? 'Upload data to begin.' : `Tracking ${workouts.length.toLocaleString()} total sets.`}
                            </p>
                        </div>
                        {isLoaded && workouts.length > 0 && activeTab !== 'dashboard' && (
                            <button className="btn-primary" onClick={() => { }}>Generate Report</button>
                        )}
                    </header>

                    <div className="content-area">
                        {!isLoaded || workouts.length === 0 ? (
                            <FileUpload />
                        ) : (
                            <>
                                {activeTab === 'dashboard' && <Dashboard />}
                                {activeTab === 'biomechanics' && <Biomechanics />}
                                {activeTab === 'periodization' && <Periodization />}
                                {activeTab === 'insights' && <Insights />}
                                {activeTab === 'predictions' && <Predictions />}
                                {activeTab === 'history' && <div style={{ color: 'var(--text-secondary)' }}>History log coming soon...</div>}
                            </>
                        )}
                    </div>

                </div>
            </main>
        </div>
    );
}

export default App;
