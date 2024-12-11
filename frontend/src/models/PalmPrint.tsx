
export const CameraMode = {
    RECOGNITION: 'RECOGNITION',
    REGISTRATION: 'REGISTRATION'
}

export const RegistrationPhases = {
    BACKGROUND_CUT: 1,
    ROI_CUT: 2,
    REGISTER_INFERENCE: 3
}

export type VideoUploadedResponse = {
    status: boolean;
    message: string;
    base64Images: string;
}

interface StudentInfo {
    id: number;
    palmPrintImages: any[]; // Use a specific type for the array elements if known
    studentCode: string;
    studentName: string;
}

export type RecognitionResult = {
    accept: boolean;
    average_occurrence_score: number;
    average_similarity_score: number;
    most_common_id: string;
    occurrence_count: number;
    score: number;
    student_info: StudentInfo;

}