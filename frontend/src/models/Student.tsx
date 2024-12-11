export const FileType = {
    BASE64: 'base64',
    FILE: 'file',
}

export type Student = {
    id: number;
    studentCode: string;
    studentName: string;
    isRegistered: boolean;
}

export type ImageFile = {
    file: File | null;
    base64: string | null;
    isSelected: boolean;
    type: string;
}

export type StudentValidationResponse = {
    isRegistered: boolean;
    statusResult: boolean;
    message: string;
}

export type ImagesResponse = {
    images: string[];
}