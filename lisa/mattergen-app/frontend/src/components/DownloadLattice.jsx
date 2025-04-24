import React, { useState } from "react";
import axios from "axios";

// const API = import.meta.env.VITE_API_URL;
const API = "http://localhost:8000"

function DownloadLattice () {
    const [latticeIds, setLatticeIds] = useState("");
    const [filename, setFilename] = useState("");

    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const parseLatticeStr = (inputStr) => {
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

    const parseFilenameStr = (inputStr) => {
        const trimmed = inputStr.trim();
        // Allow alphanumerics, underscores, dashes, and dots (no slashes or illegal chars)
        const filenameRegex = /^[\w\-\_]+$/;
        if (!filenameRegex.test(inputStr)) {
            setError('Invalid filename. Use only letters, numbers, dashes or underscores. Special characters, periods and spaces are not allowed.');
            return null;
    }
        return trimmed;
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setMessage('');
        setError('');

        const latticeIdsParsed = parseLatticeStr(latticeIds);
        const filenameParsed = parseFilenameStr(filename);

        //validate inputs
        if (!latticeIdsParsed) {
            setIsLoading(false);
            return;
        } else if (latticeIdsParsed.length === 0) {
            setError('Lattice IDs are required.');
            setIsLoading(false);
            return;
        }

        if (!filenameParsed) {
            setIsLoading(false);
            return;
        } else if (filenameParsed.length === 0) {
            setError('Filename is required.');
            setIsLoading(false);
            return;
        }

        let safeFilename = filenameParsed;
        if (!safeFilename.endsWith('.extxyz')) {
            safeFilename += '.extxyz';
        }

        const requestData = {
            lattice_ids: latticeIdsParsed,
            filename: safeFilename
        };

        try {
            const response = await axios.post(`${API}/lattices/download`, requestData, {
                headers: {
                    'Content-Type': 'application/json',
                },
                responseType: 'blob' // Expects binary content. IMPORTANT for file downloads. 
            });

            if (response.status === 200 && response.data) {
                // StreamingResponse sends a stream of bytes (file)
                // → Axios reads it as a Blob
                // → JS turns it into a downloadable file using URL.createObjectURL.
                const url = window.URL.createObjectURL(new Blob([response.data]));
                const link = document.createElement('a');
                link.href = url;
                link.setAttribute('download', safeFilename);
                document.body.appendChild(link);
                link.click();
                link.remove();
                setMessage('Lattice downloaded successfully.');
            } else {
                setMessage('Request submitted, but received an unexpected response.');
                console.error('Unexpected success response:', response);
            }
        } catch (err) {
            console.error('Error downloading lattices:', err);
            if (err.response?.data?.detail) {
                setError(`Error: ${err.response.data.detail}`);
            } else if (err.isAxiosError && !err.response) {
                setError('Network Error: Could not connect to the server. Please check your connection or the server status.');
                console.error('Network Error:', err.message);
            } else if (err.message) {
                setError(`Request Error: ${err.message}`);
            } else {
                setError('An unknown error occurred while submitting the request.');
            }
        } finally {
            setIsLoading(false);
        }
    }; 

    return (
        <div className="relative z-10 p-8 bg-slate-50 rounded-2xl shadow-[0_4px_30px_rgba(0,0,0,0.1)] border border-stone-200 max-w-3xl w-full mx-auto">
        <h2 className=" relative z-10 text-6xl font-semibold text-slate-800 mb-15 mt-5 text-center">
            Download Lattices
        </h2>

        <form onSubmit={handleSubmit} className="space-y-12">
            {/* Lattice IDs input */}
            <div>
                <label htmlFor="latticeIds" className="block text-2xl font-semibold text-emerald-700 mb-2">
                    Lattice IDs
                </label>
                {/* Example input */}
                <p className="text-md text-gray-500 mb-2">
                    Enter one or more Lattice IDs, separated by commas.
                </p> 

                <input
                    type="text"
                    id="latticeIDs"
                    value={latticeIds}
                    onChange={(e) => setLatticeIds(e.target.value)}
                    placeholder="e.g., 64f91c27a4b3d92e7ef3c5d8, 5ea4b82f1a9c3c6d2f4e9b71"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
                    disabled={isLoading}
                />
            </div>

            {/* Filename */}
            <div>
                <label htmlFor="filename" className="block text-2xl font-semibold text-emerald-700 mb-2">
                    Filename
                </label> 
                {/* Example input */}
                <p className="text-md text-gray-500 mb-2">
                    Enter a filename. The file will be saved as <strong>filename.extxyz</strong>.
                </p> 

                <input
                    type="text"
                    id="filename"
                    value={filename}
                    onChange={(e) => setFilename(e.target.value)}
                    placeholder="e.g., lattices_1-0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
                    disabled={isLoading}
                />
            </div>

            {/* Submit Button */}
            <div className='flex justify-center'>
                <button
                    type="submit"
                    className={`bg-gradient-to-r from-emerald-500 to-teal-400 hover:from-emerald-700 hover:to-teal-600 text-stone-100 font-bold text-xl py-3 px-6 rounded-2xl shadow-lg hover:shadow-xl transition duration-300 ease-in-out ${
                    isLoading
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-teal-800 hover:bg-emerald-700'
                    }`}
                    disabled={isLoading} // Disable button while loading
                >
                    {isLoading ? 'Downloading lattices...' : 'Download Lattices'}
                </button>
            </div>
        </form>

        {/* Feedback Messages */}
        {message && (
            <div className="mt-4 p-3 bg-green-100 text-green-800 border border-green-200 rounded-md text-md">
            {message}
            </div>
        )}
        {error && (
            <div className="mt-4 p-3 bg-red-100 text-red-800 border border-red-200 rounded-md text-md">
            {error}
            </div>
        )}
        </div>
  );
}

export default DownloadLattice