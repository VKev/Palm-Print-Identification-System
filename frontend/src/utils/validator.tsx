export type StudentCodeStatus = {
    isValid: boolean;
    error?: string;
}

export function checkStudentCodeFormatDetailed(studentCode: string): StudentCodeStatus {
    if (studentCode.length !== 8) {
        return { isValid: false, error: "Code must be exactly 8 characters" };
    }

    const prefix = studentCode.substring(0, 2);
    if (!['SE', 'SS', 'SA'].includes(prefix)) {
        return { isValid: false, error: "Code must start with SE, SS, or SA" };
    }

    const middleNumber = parseInt(studentCode.substring(2, 4));
    if (middleNumber < 15 || middleNumber > 20) {
        return { isValid: false, error: "Numbers must be between 15 and 20" };
    }
    const lastFour = studentCode.substring(4);
    if (!/^\d{4}$/.test(lastFour)) {
        return { isValid: false, error: "Last 4 characters must be digits" };
    }
    return { isValid: true };
}