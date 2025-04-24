import React, {useState} from "react";
import axios from "axios";

// const API = import.meta.env.VITE_API_URL;
const API = "http://localhost:8000"

function GenerateLattice () {
    const [magneticDensityStr, setMagneticDensityStr] = useState("");
    const [guidanceFactorStr, setGuidanceFactorStr] = useState("");
    const [batchSizeStr, setBatchSizeStr] = useState("");

    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    //define helper function to parse input strings
    const parseNumberList = (inputStr, fieldName) => {
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

    const handleSubmit = async (e) => {
        e.preventDefault(); // Prevent default browser form submission (page reload)
        setIsLoading(true);
        setMessage('');
        setError('');

        const magneticDensity = parseNumberList(magneticDensityStr, "Magnetic Density");
        const guidanceFactor = parseNumberList(guidanceFactorStr, "Guidance Factor");
        const batchSize = parseInt(batchSizeStr, 10);

        //validate inputs
        if (!magneticDensity) {
            setIsLoading(false);
            return;
        } else if (magneticDensity.length === 0) {
            setError('Magnetic Density is required.');
            setIsLoading(false);
            return;
        } else if (magneticDensity.some((value) => value < 0)) {
            setError('Magnetic Density values must be positive numbers.');
            setIsLoading(false);
            return;
        }

        if (!guidanceFactor) {
            setIsLoading(false);
            return;
        } else if (guidanceFactor.length === 0) {
            setError('Guidance Factor is required.');
            setIsLoading(false);
            return;
        } else if (guidanceFactor.some((value) => value < 1)) {
            setError('Guidance Factor values must be greater than 1.');
            setIsLoading(false);
            return;
        }

        if (isNaN(batchSize) || batchSize <= 0) {
            setError('Batch Size must be a positive integer.');
            setIsLoading(false);
            return;
        }
        
        // Prepare the request data
        const requestData = {
            magnetic_density: magneticDensity,
            guidance_factor: guidanceFactor,
            batch_size: batchSize
        };

        // Send the request to the backend
        try {
            // Assuming proxy setup for /api, otherwise use full URL e.g., 'http://localhost:8000/lattices'
            const response = await axios.post(`${API}/lattices`, requestData, {
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.status === 202 && response.data) {
                setMessage(`${response.data.message} ${response.data.details}`);
            } else {
                // Handle unexpected success response
                setMessage('Request submitted, but received an unexpected response.');
                console.error('Unexpected success response:', response);
            }
        } catch (err) {
            console.error('Error submitting generation request:', err);
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
    }
    
    return (
        
        <div className="relative z-10 p-8 bg-white rounded-2xl shadow-[0_4px_30px_rgba(0,0,0,0.1)] border border-stone-200 max-w-3xl w-full mx-auto">
        <h2 className=" relative z-10 text-6xl font-semibold text-slate-800 mb-15 mt-5 text-center">
            Generate Lattices
        </h2>

        <form onSubmit={handleSubmit} className="space-y-12">
            {/* Magnetic Density Input */}
            <div>
                <label htmlFor="magneticDensity" className="block text-2xl font-semibold text-emerald-700 mb-2">
                    Magnetic Density
                </label>
                {/* Example input */}
                <p className="text-md text-gray-500 mb-2">
                    Enter one or more numbers, separated by commas.
                </p> 

                <input
                    type="text"
                    id="magneticDensity"
                    value={magneticDensityStr}
                    onChange={(e) => setMagneticDensityStr(e.target.value)}
                    placeholder="e.g., 0, 0.5, 1"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
                    disabled={isLoading}
                />
            </div>

            {/* Guidance Factor Input */}
            <div>
                <label htmlFor="guidanceFactor" className="block text-2xl font-semibold text-emerald-700 mb-2">
                    Guidance Factor
                </label>
                {/* Example input */}
                <p className="text-md text-gray-500 mb-2">
                    Enter one or more numbers, separated by commas.
                </p> 

                <input
                    type="text"
                    id="guidanceFactor"
                    value={guidanceFactorStr}
                    onChange={(e) => setGuidanceFactorStr(e.target.value)}
                    placeholder="e.g., 1, 2.5, 4"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
                    disabled={isLoading}
                />
            </div>

            {/* Batch Size Input */}
            <div>
                <label htmlFor="batchSize" className="block text-2xl font-semibold text-emerald-700 mb-2">
                    Batch Size
                </label>
                {/* Example input */}
                <p className="text-md text-gray-500 mb-2">
                    Enter an integer.
                </p> 

                <input
                    type="number"
                    id="batchSize"
                    value={batchSizeStr}
                    onChange={(e) => setBatchSizeStr(e.target.value)}
                    placeholder="e.g., 8"
                    min="8"
                    step="8"
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
                    {isLoading ? 'Scheduling Task...' : 'Generate Lattices'}
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

export default GenerateLattice;