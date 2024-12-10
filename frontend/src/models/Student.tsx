
export type ImageFile = {
    file: File;
    isSelected: boolean;
}

export type StudentValidationResponse = {
    isRegistered: boolean;
    statusResult: boolean;
    message: string;
}