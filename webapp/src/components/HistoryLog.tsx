import React, { useMemo, useState } from 'react';
import { useStore } from '../lib/store';
import { Calendar, Clock, ChevronDown, ChevronUp } from 'lucide-react';
import { WorkoutSet } from '../types/workout';

export const HistoryLog: React.FC = () => {
    const workouts = useStore(state => state.filteredWorkouts);
    const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

    // Group sets by workout session (date + workoutName)
    const groupedWorkouts = useMemo(() => {
        const groups: Record<string, {
            id: string;
            date: string;
            workoutName: string;
            durationMinutes: number;
            totalVolume: number;
            totalSets: number;
            exercises: Record<string, WorkoutSet[]>;
        }> = {};

        workouts.forEach(set => {
            const workoutId = `${set.date}_${set.workoutName}`;

            if (!groups[workoutId]) {
                groups[workoutId] = {
                    id: workoutId,
                    date: set.date,
                    workoutName: set.workoutName,
                    durationMinutes: set.durationMinutes || 0,
                    totalVolume: 0,
                    totalSets: 0,
                    exercises: {}
                };
            }

            const group = groups[workoutId];
            group.totalVolume += (set.weight * set.reps);
            group.totalSets += 1;

            if (!group.exercises[set.exerciseName]) {
                group.exercises[set.exerciseName] = [];
            }
            group.exercises[set.exerciseName].push(set);
        });

        // Sort workouts descending (newest first)
        return Object.values(groups).sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
    }, [workouts]);

    const toggleExpand = (id: string) => {
        setExpandedIds(prev => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    if (groupedWorkouts.length === 0) {
        return <div style={{ color: 'var(--text-secondary)' }}>No history found for this period.</div>;
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <h3 style={{ fontSize: '1.2rem', color: 'var(--text-secondary)' }}>Workout Feed ({groupedWorkouts.length} sessions)</h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {groupedWorkouts.map((workout) => {
                    const isExpanded = expandedIds.has(workout.id);
                    const formattedDate = new Date(workout.date.replace(' ', 'T')).toLocaleString(undefined, {
                        weekday: 'short', month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit'
                    });

                    return (
                        <div key={workout.id} className="glass-panel" style={{ padding: '0', overflow: 'hidden', border: isExpanded ? '1px solid rgba(139, 92, 246, 0.3)' : undefined }}>
                            {/* Header (Clickable snippet) */}
                            <div
                                onClick={() => toggleExpand(workout.id)}
                                className="hover-highlight"
                                style={{
                                    padding: '20px 24px',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    cursor: 'pointer',
                                    background: isExpanded ? 'rgba(255,255,255,0.02)' : 'transparent'
                                }}
                            >
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                    <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                                        {workout.workoutName || 'Unnamed Workout'}
                                    </div>
                                    <div style={{ display: 'flex', gap: '16px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Calendar size={14} /> {formattedDate}</span>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Clock size={14} /> {workout.durationMinutes}m duration</span>
                                    </div>
                                </div>

                                <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
                                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                                        <span style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-primary)' }}>
                                            {workout.totalVolume.toLocaleString()} lbs
                                        </span>
                                        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                            {workout.totalSets} sets • {Object.keys(workout.exercises).length} exercises
                                        </span>
                                    </div>
                                    <button style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', padding: '4px' }}>
                                        {isExpanded ? <ChevronUp size={24} /> : <ChevronDown size={24} />}
                                    </button>
                                </div>
                            </div>

                            {/* Expanded Content (Sets Table) */}
                            {isExpanded && (
                                <div style={{ padding: '0 24px 24px 24px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                                    {Object.entries(workout.exercises).map(([exerciseName, sets]) => (
                                        <div key={exerciseName} style={{ marginTop: '20px' }}>
                                            <h4 style={{ fontSize: '1rem', color: 'var(--text-primary)', marginBottom: '12px', paddingBottom: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                                {exerciseName}
                                            </h4>

                                            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(50px, auto) 1fr 1fr 1fr minmax(100px, auto)', gap: '12px', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '8px', padding: '0 8px' }}>
                                                <div>Set</div>
                                                <div>Lbs</div>
                                                <div>Reps</div>
                                                <div>Est. 1RM</div>
                                                <div style={{ textAlign: 'right' }}>Distance/Time</div>
                                            </div>

                                            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                                {sets.map((set, idx) => (
                                                    <div key={set.id} style={{ display: 'grid', gridTemplateColumns: 'minmax(50px, auto) 1fr 1fr 1fr minmax(100px, auto)', gap: '12px', padding: '8px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', fontSize: '0.9rem', alignItems: 'center' }}>
                                                        <div style={{ color: 'var(--text-secondary)' }}>{set.setOrder || idx + 1}</div>
                                                        <div style={{ fontWeight: 600 }}>{set.weight}</div>
                                                        <div style={{ fontWeight: 600 }}>{set.reps}</div>
                                                        <div style={{ color: 'var(--text-secondary)' }}>{set.e1rmBrzycki || '-'}</div>
                                                        <div style={{ textAlign: 'right', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                                                            {set.distance ? `${set.distance}` : ''} {set.seconds ? `${set.seconds}s` : ''}
                                                            {!set.distance && !set.seconds && '-'}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
