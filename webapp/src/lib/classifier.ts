interface ClassificationResult {
    muscleGroup: string;
    category: string;
    isUnilateral: boolean;
    confidence: 'high' | 'medium' | 'low' | 'unknown';
}

const MUSCLE_GROUPS = {
    Chest: { keywords: ['bench press', 'chest press', 'chest fly', 'pec deck', 'dip', 'push up', 'pushup', 'crossover'], equipment: ['pec', 'chest'] },
    Back: { keywords: ['row', 'pull up', 'pull-up', 'pulldown', 'lat', 'shrug', 'deadlift', 'back extension', 'good morning', 'face pull', 'reverse fly'], equipment: ['lat', 'back', 'cable row'] },
    Shoulders: { keywords: ['overhead press', 'military press', 'lateral raise', 'front raise', 'arnold press', 'rear delt', 'upright row', 'shoulder press', 'shoulder swing', 'y raise'], equipment: ['shoulder', 'delt'] },
    Biceps: { keywords: ['bicep', 'curl', 'preacher'], equipment: ['bicep'] },
    Triceps: { keywords: ['tricep', 'extension', 'skullcrusher', 'pushdown', 'kickback'], equipment: ['tricep'] },
    Legs: { keywords: ['squat', 'leg press', 'leg extension', 'leg curl', 'lunge', 'bulgarian split', 'hack squat', 'calf'], equipment: ['leg', 'quad', 'hamstring', 'glute'] },
    Core: { keywords: ['crunch', 'plank', 'sit up', 'sit-up', 'ab', 'russian twist', 'leg raise'], equipment: ['ab', 'core'] }
};

export class ExerciseClassifier {
    public static classify(exerciseName: string): ClassificationResult {
        const rawName = exerciseName.toLowerCase();

        // Default fallback
        const result: ClassificationResult = {
            muscleGroup: 'Other',
            category: 'accessory',
            isUnilateral: rawName.includes('single') || rawName.includes('unilateral') || rawName.includes('one arm') || rawName.includes('one-arm') || rawName.includes('alternating') || rawName.includes('dumbbell') || rawName.includes('iso-lateral'),
            confidence: 'low'
        };

        // Advanced Dip Classification logic port
        if (rawName.includes('dip')) {
            if (rawName.includes('chest') || rawName.includes('gironda')) {
                return { ...result, muscleGroup: 'Chest', category: 'horizontal_push', confidence: 'high' };
            }
            if (rawName.includes('tricep') || rawName.includes('bench dip')) {
                return { ...result, muscleGroup: 'Triceps', category: 'isolation', confidence: 'high' };
            }
            return { ...result, muscleGroup: 'Chest', category: 'horizontal_push', confidence: 'medium' }; // Default standard dips to chest
        }

        // Pattern matching logic
        for (const [muscle, data] of Object.entries(MUSCLE_GROUPS)) {
            if (data.keywords.some(k => rawName.includes(k))) {
                result.muscleGroup = muscle;
                result.confidence = 'medium';
                break;
            }
        }

        // Categorization logic
        if (result.muscleGroup === 'Chest') result.category = 'horizontal_push';
        else if (result.muscleGroup === 'Shoulders') result.category = 'vertical_push';
        else if (result.muscleGroup === 'Back' && (rawName.includes('pull') || rawName.includes('chin'))) result.category = 'vertical_pull';
        else if (result.muscleGroup === 'Back' && rawName.includes('row')) result.category = 'horizontal_pull';
        else if (result.muscleGroup === 'Legs' && (rawName.includes('squat') || rawName.includes('press'))) result.category = 'lower_compound';
        else result.category = 'isolation';

        if (result.muscleGroup !== 'Other') {
            result.confidence = 'medium';
        }

        return result;
    }
}
