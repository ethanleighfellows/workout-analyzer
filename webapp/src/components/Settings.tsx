import React, { useState } from 'react';
import { useStore } from '../lib/store';
import { Trash2, Save, Download, AlertTriangle, User, Target } from 'lucide-react';

export const Settings: React.FC = () => {
    const { clearData, workouts, summary, athleteWeight, athleteGender, setAthleteProfile } = useStore();
    const [isConfirmingClear, setIsConfirmingClear] = useState(false);

    const [tempWeight, setTempWeight] = useState(athleteWeight ? athleteWeight.toString() : '');
    const [tempGender, setTempGender] = useState<'M' | 'F'>(athleteGender);
    const [isSaved, setIsSaved] = useState(false);

    const handleExport = () => {
        // Simple JSON export for now
        const dataStr = JSON.stringify({ workouts, summary }, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);

        const exportFileDefaultName = `workout_analyzer_backup_${new Date().toISOString().split('T')[0]}.json`;

        let linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
    };

    const handleClear = () => {
        if (isConfirmingClear) {
            clearData();
            setIsConfirmingClear(false);
        } else {
            setIsConfirmingClear(true);
        }
    };

    const handleSaveProfile = () => {
        setAthleteProfile(tempWeight ? Number(tempWeight) : null, tempGender);
        setIsSaved(true);
        setTimeout(() => setIsSaved(false), 2000);
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', maxWidth: '800px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <h3 style={{ fontSize: '1.2rem', color: 'var(--text-secondary)' }}>App Settings & Data Management</h3>
            </div>

            <div className="glass-panel" style={{ padding: '24px' }}>
                <h4 style={{ fontSize: '1.1rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <User size={20} /> Athlete Profile
                </h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '24px' }}>
                    Input your bodyweight to dynamically calculate Wilks & DOTS powerlifting coefficient scores across your historical lifts.
                </p>

                <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Bodyweight (lbs)</label>
                        <input
                            type="number"
                            value={tempWeight}
                            onChange={(e) => setTempWeight(e.target.value)}
                            placeholder="e.g. 185"
                            style={{
                                background: 'rgba(255,255,255,0.05)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                padding: '10px 14px',
                                borderRadius: '8px',
                                color: 'white',
                                width: '120px'
                            }}
                        />
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Scoring Gender</label>
                        <select
                            value={tempGender}
                            onChange={(e) => setTempGender(e.target.value as 'M' | 'F')}
                            style={{
                                background: 'rgba(255,255,255,0.05)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                padding: '10px 14px',
                                borderRadius: '8px',
                                color: 'white',
                                width: '120px'
                            }}
                        >
                            <option value="M">Male</option>
                            <option value="F">Female</option>
                        </select>
                    </div>

                    <button
                        onClick={handleSaveProfile}
                        style={{
                            background: isSaved ? 'rgba(34, 197, 94, 0.2)' : 'rgba(139, 92, 246, 0.2)',
                            color: isSaved ? '#22c55e' : '#8b5cf6',
                            border: isSaved ? '1px solid rgba(34, 197, 94, 0.3)' : '1px solid rgba(139, 92, 246, 0.3)',
                            padding: '10px 20px',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            fontWeight: 600,
                            transition: 'all 0.2s'
                        }}
                    >
                        {isSaved ? <Target size={18} /> : <Save size={18} />}
                        {isSaved ? 'Saved!' : 'Save Profile'}
                    </button>
                </div>
            </div>

            <div className="glass-panel" style={{ padding: '24px' }}>
                <h4 style={{ fontSize: '1.1rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Save size={20} /> Data Storage
                </h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '24px' }}>
                    Your data is stored locally in your browser's IndexedDB. It never leaves your device. You are currently tracking <strong>{workouts.length} sets</strong> locally.
                </p>

                <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                    <button
                        onClick={handleExport}
                        style={{
                            background: 'rgba(255, 255, 255, 0.05)',
                            color: 'var(--text-primary)',
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            padding: '10px 20px',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            fontWeight: 600
                        }}
                    >
                        <Download size={18} /> Export JSON Backup
                    </button>

                    <button
                        onClick={handleClear}
                        style={{
                            background: isConfirmingClear ? 'rgba(239, 68, 68, 0.2)' : 'transparent',
                            color: isConfirmingClear ? '#ef4444' : 'var(--text-secondary)',
                            border: isConfirmingClear ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid rgba(255, 255, 255, 0.1)',
                            padding: '10px 20px',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            fontWeight: 600
                        }}
                    >
                        {isConfirmingClear ? <AlertTriangle size={18} /> : <Trash2 size={18} />}
                        {isConfirmingClear ? 'Click again to permanently delete data' : 'Clear Local Data'}
                    </button>
                    {isConfirmingClear && (
                        <button
                            onClick={() => setIsConfirmingClear(false)}
                            style={{ background: 'transparent', color: 'var(--text-secondary)', border: 'none', cursor: 'pointer', padding: '10px' }}
                        >
                            Cancel
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};
