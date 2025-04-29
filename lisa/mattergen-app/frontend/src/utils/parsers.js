//helper for download
export const parseLatticeStr = (inputStr) => {
    const parsedInput = inputStr
                            .split(',')
                            .map(s => s.trim())
                            .filter(s => s !== '')
    const latticeRegex = /^[0-9a-fA-F]{24}$/;
    // MongoDB ObjectId validation
    const invalidInstances = parsedInput.filter(s => !latticeRegex.test(s));
    if (invalidInstances.length > 0) {
        setError(`Invalid input: ${invalidInstances.join(', ')}. All values must be 24-character hexadecimal strings.`);
        return null;
    }
    return parsedInput
};

export const parseFilenameStr = (inputStr) => {
    const trimmed = inputStr.trim();
    // Allow alphanumerics, underscores, dashes, and dots (no slashes or illegal chars)
    const filenameRegex = /^[\w\-\_]+$/;
    if (!filenameRegex.test(inputStr)) {
        setError('Invalid filename. Use only letters, numbers, dashes or underscores. Special characters, periods and spaces are not allowed.');
        return null;
}
    return trimmed;
}

//helper for generate
export const parseNumberList = (inputStr, fieldName) => {
    const parsedInput = inputStr
                            .split(',')
                            .map(s => s.trim())
                            .filter(s => s !== '')
                            .map(Number);

    const hasInvalid = parsedInput.some(n => isNaN(n));
    if (hasInvalid) {
        setError(`Invalid input in ${fieldName}: All values must be numbers.`);
        return null;
    }
    return parsedInput
};

//helper for retrieve by filter
export const parseAtomsList = (inputStr) => {
    const parsedInput = inputStr
        .split(',')
        .map(s => s.trim())
        .filter(s => s !== '');

    const atomsRegex = /^[a-zA-Z]{1,2}$/;
    const invalidAtoms = parsedInput.filter(s => !atomsRegex.test(s));
    if (invalidAtoms.length > 0) {
        setError(`Invalid atom symbols: ${invalidAtoms.join(", ")}. Each must be 1–2 letters.`);
        return null;
    }
    return parsedInput
}