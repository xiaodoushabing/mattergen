import React, {useState} from "react";
import axios from "axios";

function GenerateLattice () {
    const [magneticDensityStr, setMagneticDensityStr] = useState("0");
    const [guidanceFactorStr, setGuidanceFactorStr] = useState("0");
    const [batchSizeStr, setBatchSizeStr] = useState("8");

    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    //define helper function to parse input strings
    const parseNumberList = (inputStr) => {
        if (!inputStr) return [];
        return inputStr
            .split(',')
            .map(s => s.trim())
            .filter(s => s !== '')
            .map(Number)
            .filter(n => !isNaN(n));
    };

    const handleSubmit = async (e) => {
        e.preventDefault(); // Prevent default browser form submission (page reload)
        setIsLoading(true);
        setMessage('');
        setError('');

        const magneticDensity = parseNumberList(magneticDensityStr);
        const guidanceFactor = parseNumberList(guidanceFactorStr);
        const batchSize = parseInt(batchSizeStr, 10);

        //validate inputs
        if (magneticDensity.length === 0) {
            setError('Magnetic Density is required');
            setIsLoading(false);
            return;
        }
        if (guidanceFactor.length === 0) {
            setError('Guidance Factor is required');
            setIsLoading(false);
            return;
        }

        if (isNaN(batchSize) || batchSize <= 0) {
            setError('Batch Size must be a positive whole number');
            setIsLoading(false);
            return;
        }

        // Prepare the request data
        const requestData = {
            magneticDensity: magneticDensity,
            guidanceFactor: guidanceFactor,
            batchSize: batchSize
        };

        // Send the request to the backend
        try {
            // Assuming proxy setup for /api, otherwise use full URL e.g., 'http://localhost:8000/lattices'
            const response = await axios.post('/api/generate-lattice', requestData, {
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.status === 202 && response.data) {
                setMessage(`${response.data.message} ${response.data.details} || ''`);
            } else {
                // Handle unexpected success response
                setMessage('Request submitted, but received an unexpected response.');
                console.error('Unexpected success response:', response);
            }
        } catch (err) {
            console.error('Error submitting generation requet:', err);
            if (err.response && err.response.data && err.response.data.detail) {
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
    }
    
    return (
        <div className="p-6 bg-white rounded-lg shadow-md max-w-lg mx-auto mt-10">
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">Generate Lattices</h2>

        <form onSubmit={handleSubmit} className="space-y-4">
            {/* Magnetic Density Input */}
            <div>
            <label htmlFor="magneticDensity" className="block text-sm font-medium text-gray-700 mb-1">
                Magnetic Density (comma-separated numbers)
            </label>
            <input
                type="text"
                id="magneticDensity"
                value={magneticDensityStr}
                onChange={(e) => setMagneticDensityStr(e.target.value)}
                placeholder="e.g., 0, 0.5, 1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
                disabled={isLoading} // Disable input while loading
            />
            </div>

            {/* Guidance Factor Input */}
            <div>
            <label htmlFor="guidanceFactor" className="block text-sm font-medium text-gray-700 mb-1">
                Guidance Factor (comma-separated numbers)
            </label>
            <input
                type="text"
                id="guidanceFactor"
                value={guidanceFactorStr}
                onChange={(e) => setGuidanceFactorStr(e.target.value)}
                placeholder="e.g., 1, 2, 4"
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
                disabled={isLoading}
            />
            </div>

            {/* Batch Size Input */}
            <div>
            <label htmlFor="batchSize" className="block text-sm font-medium text-gray-700 mb-1">
                Batch Size
            </label>
            <input
                type="number"
                id="batchSize"
                value={batchSizeStr}
                onChange={(e) => setBatchSize(parseInt(e.target.value, 10) || 0)} // Parse to integer
                min="1" // Set minimum value
                step="1" // Allow only whole numbers
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
                disabled={isLoading}
            />
            </div>

            {/* Submit Button */}
            <div>
            <button
                type="submit"
                className={`w-full px-4 py-2 text-white font-semibold rounded-md transition duration-200 ease-in-out ${
                isLoading
                    ? 'bg-gray-400 cursor-not-allowed' // Style for loading state
                    : 'bg-emerald-600 hover:bg-emerald-700' // Normal style
                }`}
                disabled={isLoading} // Disable button while loading
            >
                {isLoading ? 'Scheduling Task...' : 'Generate Lattices'}
            </button>
            </div>
        </form>

        {/* Feedback Messages */}
        {message && (
            <div className="mt-4 p-3 bg-green-100 text-green-800 border border-green-200 rounded-md text-sm">
            {message}
            </div>
        )}
        {error && (
            <div className="mt-4 p-3 bg-red-100 text-red-800 border border-red-200 rounded-md text-sm">
            {error}
            </div>
        )}
        </div>
  );
}

export default GenerateLattice;