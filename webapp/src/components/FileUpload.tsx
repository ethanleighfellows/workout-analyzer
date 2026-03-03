import React, { useRef, useState } from 'react';
import { UploadCloud } from 'lucide-react';
import { parseStrongCSV } from '../lib/parser';
import { useStore } from '../lib/store';

export const FileUpload: React.FC = () => {
    const [isDragActive, setIsDragActive] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const importData = useStore((state) => state.importData);
    const isLoading = useStore((state) => state.isLoading);
    const [localError, setLocalError] = useState<string | null>(null);

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            await processFile(file);
        }
    };

    const processFile = async (file: File) => {
        try {
            setLocalError(null);
            if (!file.name.endsWith('.csv')) {
                throw new Error('Please upload a valid CSV file from Strong App.');
            }
            const sets = await parseStrongCSV(file);
            await importData(sets);
        } catch (err: any) {
            setLocalError(err.message);
        }
    };

    const onDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragActive(true);
    };

    const onDragLeave = () => {
        setIsDragActive(false);
    };

    const onDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            await processFile(e.dataTransfer.files[0]);
        }
    };

    return (
        <div
            className={`glass-panel ${isDragActive ? 'drag-active' : ''}`}
            style={{
                padding: '40px',
                textAlign: 'center',
                borderStyle: 'dashed',
                borderWidth: '2px',
                borderColor: isDragActive ? 'var(--accent-primary)' : 'var(--border-color)',
                transition: 'all 0.3s ease',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '300px'
            }}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
        >
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".csv"
                style={{ display: 'none' }}
            />

            {isLoading ? (
                <div style={{ animation: 'pulse 1.5s infinite' }}>
                    <UploadCloud size={48} color="var(--accent-primary)" style={{ marginBottom: '20px' }} />
                    <h3 style={{ fontSize: '1.5rem', marginBottom: '10px' }}>Analyzing Workout Data...</h3>
                </div>
            ) : (
                <>
                    <UploadCloud size={48} color="var(--text-secondary)" style={{ marginBottom: '20px' }} />
                    <h3 style={{ fontSize: '1.5rem', marginBottom: '10px' }}>Upload Strong Workout Data</h3>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: '20px', maxWidth: '400px' }}>
                        Drag and drop your Strong App CSV export here, or click to browse. The data is processed 100% locally and stored securely in your browser.
                    </p>
                    <button className="btn-primary" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                        Select CSV File
                    </button>
                </>
            )}

            {localError && (
                <div style={{ marginTop: '20px', color: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', padding: '10px 20px', borderRadius: '8px' }}>
                    {localError}
                </div>
            )}
        </div>
    );
};
