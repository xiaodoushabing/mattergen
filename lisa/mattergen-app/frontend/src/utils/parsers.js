import { data } from "react-router-dom";

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
        return {
            data: null,
            error: `Invalid input: ${invalidInstances.join(', ')}. All values must be 24-character hexadecimal strings.`
        };
    }
    return  { data: parsedInput, error: null }
};

export const parseFilenameStr = (inputStr) => {
    const trimmed = inputStr.trim();
    const filenameRegex = /^[\w\-\_]+$/;
    if (!filenameRegex.test(trimmed)) {
        return {
            data: null,
            error: 'Invalid filename. Use only letters, numbers, dashes or underscores. Special characters, periods and spaces are not allowed.'
        };
    }
    return { data: trimmed, error: null };
};

export const parseNumberList = (inputStr, fieldName) => {
    const parsedInput = inputStr
        .split(',')
        .map(s => s.trim())
        .filter(s => s !== '')
        .map(Number);

    const hasInvalid = parsedInput.some(n => isNaN(n));
    if (hasInvalid) {
        return {
            data: null,
            error: `Invalid input in ${fieldName}: All values must be numbers.`
        };
    }
    return { data: parsedInput, error: null };
};

export const parseAtomsList = (inputStr) => {
    const parsedInput = inputStr
        .split(',')
        .map(s => s.trim())
        .filter(s => s !== '');

    const atomsRegex = /^[a-zA-Z]{1,2}$/;
    const invalidAtoms = parsedInput.filter(s => !atomsRegex.test(s));

    if (invalidAtoms.length > 0) {
        return {
            data: null,
            error: `Invalid atom symbols: ${invalidAtoms.join(', ')}. Each must be 1–2 letters.`
        };
    }
    return { data: parsedInput, error: null };
};