export const base64ToFile = (base64String: string, filename: string): File => {
  // Remove data URL prefix if present
  const base64Content = base64String.replace(/^data:.*?;base64,/, '');
  
  // Convert base64 to binary
  const byteCharacters = atob(base64Content);
  const byteArrays = [];
  
  // Split into chunks and convert to TypedArray
  for (let offset = 0; offset < byteCharacters.length; offset += 512) {
    const slice = byteCharacters.slice(offset, offset + 512);
    const byteNumbers = new Array(slice.length);
    
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }
    
    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }

  // Create blob from binary data
  const blob = new Blob(byteArrays);
  
  // Create and return File object
  return new File([blob], filename);
};