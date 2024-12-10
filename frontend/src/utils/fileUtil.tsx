
export function base64ToFile(base64: string, fileName: string): File {
    // Split the Base64 string into the data part and the MIME type
    const [prefix, base64Data] = base64.split(',');

    // Decode the Base64 string into a byte array
    const byteCharacters = atob(base64Data);

    // Create a Uint8Array to hold the byte data
    const byteArray = new Uint8Array(byteCharacters.length);

    // Fill the Uint8Array with byte values
    for (let i = 0; i < byteCharacters.length; i++) {
        byteArray[i] = byteCharacters.charCodeAt(i);
    }

    // Create a Blob from the byte array and MIME type
    const blob = new Blob([byteArray], { type: getMimeType(prefix) });

    // Convert the Blob to a File
    const file = new File([blob], fileName, { type: getMimeType(prefix) });

    return file;
}

function getMimeType(prefix: string): string {
    const mimeTypeMatch = prefix.match(/^data:(.*?);/);
    return mimeTypeMatch ? mimeTypeMatch[1] : 'application/octet-stream';
}
