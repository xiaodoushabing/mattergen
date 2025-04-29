import React, { useState } from "react";
import axios from "axios";
import { parseLatticeStr } from "../utils/parsers";

// const API = import.meta.env.VITE_API_URL;
const API = "http://localhost:8000"

function RetrieveLatticeById() {
    const [latticeId, setLatticeId] = useState("");
    const [error, setError] = useState("");
    const [message, setMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [results, setResults] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setMessage('');
        setError('');

        const latticeIdParsed = parseLatticeStr(latticeId);

        //validate inputs
        if (!latticeIdParsed) {
            setIsLoading(false);
            return;
        } else if (latticeIdParsed.length === 0) {
            setError('Lattice ID is required.');
            setIsLoading(false);
            return;
        } else if (latticeIdParsed.length > 1) {
            setError('Only one lattice ID is allowed.');
            setIsLoading(false);
            return;
        }

        try {
            const response = await axios.get(`${API}/lattice/${latticeIdParsed}`, {
            headers: {
                'Content-Type': 'application/json',
            }});
            
            if (response.status === 200 && response.data) {
                setResults(response.data);
                setMessage('Lattice data retrieved successfully.');
            } else {
                setMessage('Request submitted, but received an unexpected response.');
                console.error('Unexpected success response:', response);
            }

        } catch (error) {
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
    
    const InfoField = ({ label, value }) => (
        <p className="mb-1">
            <strong>{label}</strong>
            <span className="block ml-4 text-gray-700">{value}</span>
        </p>
    );

    return (
    <div className="relative z-10 p-8 bg-slate-50 rounded-2xl shadow-[0_4px_30px_rgba(0,0,0,0.1)] border border-stone-200 max-w-4xl w-full mx-auto">
        <form onSubmit={handleSubmit} className="space-y-10">
        {/* Lattice ID input */}
        <div>
            <label htmlFor="latticeId" className="block text-lg font-semibold text-emerald-700 mb-1">
            Lattice ID
            </label>
            <p className="text-sm text-gray-500 mb-2">
            Enter only <strong>one</strong> Lattice ID (a 24-character hexadecimal).
            </p> 
            <input
            type="text"
            id="latticeID"
            value={latticeId}
            onChange={(e) => setLatticeId(e.target.value)}
            placeholder="e.g., 64f91c27a4b3d92e7ef3c5d8"
            maxLength={24}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
            disabled={isLoading}
            />
        </div>

        <div className="flex justify-center">
            <button
            type="submit"
            className={`bg-gradient-to-r from-emerald-500 to-teal-400 hover:from-emerald-700 hover:to-teal-600 text-stone-100 font-bold text-lg py-3 px-6 rounded-2xl shadow-lg hover:shadow-xl transition duration-300 ease-in-out ${
                isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-teal-800 hover:bg-emerald-700'
            }`}
            disabled={isLoading}
            >
            {isLoading ? 'Retrieving lattice...' : 'Retrieve by ID'}
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

        {/* Results Display */}
        {results && (
        <div className="mt-8">
            <h3 className="text-2xl font-bold text-slate-700 mb-6">Results:</h3>

            <div className="p-4 border border-gray-200 rounded-lg bg-gray-100 flex justify-between items-start text-sm">
            <div>
                <InfoField label="Number of atoms:" value={`${results.no_of_atoms}`} />
                <p className="mb-1">
                <strong>Elements:</strong>
                {Object.entries(results.atoms_list).map(([element, count]) => (
                    <span key={element} className="block ml-4">
                    <span className="font-semibold text-emerald-700">{element}</span>: 
                    {' '}
                    <span className="text-gray-700">{count}</span>
                    </span>
                ))}
                </p>

                <InfoField label="Guidance Factor:" value={results.guidance_factor} />
                <InfoField label="Magnetic Density:" value={results.magnetic_density} />
                <InfoField label="Periodic Boundary Condition:" value={results.pbc.split('').join(' ')} />

                <p className="mb-1">
                <strong>Cell Parameters:</strong>
                <span className="block ml-4 text-gray-700">
                    {'['}
                    {results.cell_parameters.map((param, index) => (
                    <span key={index}>
                        {param.toFixed(3)}
                        {index < results.cell_parameters.length - 1 ? ', ' : ''}
                    </span>
                    ))}
                    {']'}
                </span>
                </p>

                <p className="mb-1">
                <strong>Atom Coordinates:</strong>
                {Object.entries(results.atoms).map(([atomIdx, atomInfo]) => {
                    const [element, coords] = Object.entries(atomInfo)[0];
                    return (
                    <span key={atomIdx} className="block ml-4">
                        <span className="font-semibold text-emerald-700">{element}</span>: 
                        {' '}
                        <span className="text-gray-700">
                        [{coords.map(coord => coord.toFixed(2)).join(', ')}]
                        </span>
                    </span>
                    );
                })}
                </p>

                {results.ms_predictions?.energy && (
                <InfoField label="Energy:" value={results.ms_predictions.energy.toFixed(3)} />
                )}
            </div>

            {/* ID and Index */}
            <div className="text-right text-xs text-gray-500 ml-4">
                <p className="font-semibold mb-1">
                ID: {results._id}
                </p>
                <p>Index: {results.lattice_index}</p>
            </div>
            </div>
        </div>
        )}
    </div>
    );
}

export default RetrieveLatticeById;