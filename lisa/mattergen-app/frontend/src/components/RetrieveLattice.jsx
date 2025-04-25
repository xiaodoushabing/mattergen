import React, { useState } from "react";
import axios from "axios";
import FilterInputRow from "./FilterInputRow";

// const API = import.meta.env.VITE_API_URL;
const API = "http://localhost:8000"

function RetrieveLattice () {
    // Define configuration for each filter field
    const filterFieldsConfig = [
    { name: 'Limit', label: 'limit', defaultOp: 'eq', defaultValue: '10', placeholder: 'e.g., 10', numberType: 'integer', min: 1 },
    { name: 'LatticeIndex', label: 'latticeIndex', defaultOp: 'eq', defaultValue: '', placeholder: 'e.g., 5', numberType: 'integer', min: 1 },
    { name: 'GuidanceFactor', label: 'guidanceFactor', defaultOp: 'eq', defaultValue: '', placeholder: 'e.g., 4.0', numberType: 'float', min: 1 },
    { name: 'MagneticDensity', label: 'magneticDensity', defaultOp: 'eq', defaultValue: '', placeholder: 'e.g., 0.5', numberType: 'float', min: 0 },
    { name: 'NoOfAtoms', label: 'numberOfAtoms', defaultOp: 'eq', defaultValue: '', placeholder: 'e.g., 12', numberType: 'integer', min: 1 },
    { name: 'Energy', label: 'energy', defaultOp: 'eq', defaultValue: '', placeholder: 'e.g., -330.9', numberType: 'float' },
];

    // Valid operator options
    const opOptions = ["eq", "neq", "lt", "gt", "lte", "gte"];

    const [filters, setFilters] = useState(() => {
        const initialState = {};
        filterFieldsConfig.forEach(field => {
            initialState[field.name] = {
                operator: field.defaultOp,
                value: field.defaultValue,
            };
        });
        
        initialState.AtomsList = '';
        return initialState;
    });


    const [error, setError] = useState('');
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [nextPageLastId, setNextPageLastId] = useState(null);
    const [results, setResults] = useState([]);

    // Handlers to update the filters state
    const handleFilterChange = (fieldName, key, value) => {
        setFilters(prevFilters => ({
            ...prevFilters,
            [fieldName]: {
                ...prevFilters[fieldName],
                [key]: value
            }
        }));
    };

    const handleAtomsListChange = (value) => {
         setFilters(prevFilters => ({
            ...prevFilters,
            AtomsList: value
        }));
    };

    // validate atoms list
    const parseAtomsList = (inputStr) => {
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

    const handleSubmit = async (e, loadMore = false) => {
        e.preventDefault();

        // Reset state if not loading more
        if (!loadMore) {
            setResults([]);
            setNextPageLastId(null);
        }

        setIsLoading(true);
        setMessage('');
        setError('');

        // Build Request Payload
        const requestPayload = {
            limit: parseInt(filters.Limit.value),
        };
        
        const atoms_list = parseAtomsList(filters.AtomsList)
        if (atoms_list && atoms_list.length > 0) {
            requestPayload.atoms_list = atoms_list;
        }

        filterFieldsConfig.forEach(field => {
            const filterState = filters[field.name];
            const filterValue = filterState.value;
            const filterOp = filterState.op;
            const filterType = field.numberType;

            if (!opOptions.includes(filterOp)) {
                setError(`Invalid operator for ${field.label}.`);
                setIsLoading(false);
                return;
            }

            if (filterValue) {
                valueNum = (filterType === 'float') ? parseFloat(filterValue) : parseInt(filterValue);

                if (isNaN(valueNum)) {
                    setError(`${key.replace(/([A-Z])/g, ' $1').trim()} must be a valid ${filterType}.`);
                    setIsLoading(false);
                    return;
                }
                
                if (field.min !== undefined && valueNum < field.min) {
                     setError(`${field.label} must be at least ${field.min}.`);
                     setIsLoading(false);
                     return;
                }

                // map state key to backend snakecase format
                const apiFieldName = field.name.replace(/([A-Z])/g, '_$1').toLowerCase().replace(/^_/, '');
                requestPayload[apiFieldName] = { value: valueNum, op: filterOp};
            }
        })

        // add last_id for pagination if loading more
        // Axios will automatically serialize params and append them to the URL
        // e.g., /retrieve/lattices?last_id=abc123

        const params = loadMore && nextPageLastId ? { last_id: nextPageLastId } : {};

        console.log("Sending Retrieve Request:", requestPayload, "Params:", params);    

        // make API request
        try {
            const response = await axios.post(`${API}/lattices/filter`, requestPayload, { params });
            if (response.status === 200 && response.data?.lattices) {
                setResults(prevResults => loadMore ? [...prevResults, ...response.data.lattices] : response.data.lattices);
                setNextPageLastId(response.data.next_page_last_id);
                if (!loadMore && response.data.lattices.length === 0) {
                    setMessage(`No lattices found matching your criteria.`);
                } else if (!loadMore && response.data.lattices.length > 0) {
                    setMessage(`Successfully retrieved ${response.data.lattices.length} lattices.`);
                } else if (loadMore && response.data.lattices.length > 0) {
                    setMessage(`Successfully retrieved ${response.data.lattices.length} more lattices.`);
                } else {
                    setMessage('No more lattices available.');
                }
            } else {
                setError('Failed to retrieve lattices. Please try again.');
                setResults([]);
                setNextPageLastId(null);
            }
        } catch (err) {
            console.error("Error retrieving lattices:", err);
            const errorDetail = err.response?.data?.detail || err.message || 'An unknown error occurred.';
            setError(`Error: ${errorDetail}`);
            setResults([]);
            setNextPageLastId(null);
            
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="relative z-10 p-8 bg-slate-50 rounded-2xl shadow-[0_4px_30px_rgba(0,0,0,0.1)] border border-stone-200 max-w-2xl w-full mx-auto">
        <h2 className=" relative z-10 text-3xl font-semibold text-slate-800 mb-10 mt-5 text-center">
            Retrieve Lattices
        </h2>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-6">
                {/* Render filter rows dynamically */}
                {filterFieldsConfig.map(field => (
                    <FilterInputRow
                        key={field.name}
                        fieldConfig={field}
                        filterState={filters[field.name]}
                        onChange={handleFilterChange}
                        isLoading={isLoading}
                    />
                ))}

                {/* Atoms List Input (Separate) */}
                <div>
                    <label htmlFor="atomsList" className="block text-lg font-medium text-gray-700">
                        Atoms List (Optional)
                    </label>
                    <p className="mt-1 text-sm text-gray-500">
                        Filter by elements present. Enter symbols separated by commas (e.g., Fe, O, Pt).
                    </p>
                    <input
                        type="text"
                        id="atomsList"
                        value={filters.AtomsList}
                        onChange={(e) => handleAtomsListChange(e.target.value)}
                        placeholder="e.g., Fe, O"
                        className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
                        disabled={isLoading}
                    />
                </div>

                {/* Submit Button */}
                <div className="pt-4"> {/* Added padding top */}
                    <button
                        type="submit"
                        className={`w-full px-4 py-3 text-white font-semibold rounded-lg transition duration-200 ease-in-out text-base ${
                        isLoading
                            ? 'bg-gray-400 cursor-not-allowed'
                            : 'bg-emerald-600 hover:bg-emerald-700'
                        }`}
                        disabled={isLoading}
                    >
                        {isLoading ? 'Retrieving...' : 'Retrieve Lattices'}
                    </button>
                </div>
            </form>

            {/* Feedback Messages */}
            {/* ... (message and error display remains the same) ... */}
             {message && !error && ( // Show message only if no error
                <div className="mt-6 p-4 bg-blue-100 text-blue-800 border border-blue-200 rounded-lg text-sm">
                {message}
                </div>
            )}
            {error && (
                <div className="mt-6 p-4 bg-red-100 text-red-800 border border-red-200 rounded-lg text-sm">
                {error}
                </div>
            )}


            {/* Results Display */}
            {/* ... (results display logic remains the same) ... */}
             <div className="mt-8">
                {results.length > 0 && (
                    <h3 className="text-xl font-semibold text-slate-700 mb-4">Results:</h3>
                )}
                <div className="space-y-4">
                    {results.map((lattice) => (
                        <div key={lattice.id} className="p-4 border border-gray-200 rounded-md bg-gray-50 text-sm">
                            <p><strong>ID:</strong> {lattice.id}</p>
                            <p><strong>Index:</strong> {lattice.lattice_index}, <strong>Atoms:</strong> {lattice.no_of_atoms}, <strong>Elements:</strong> {Object.keys(lattice.atoms_list).join(', ')}</p>
                            {lattice.ms_predictions?.energy && (
                                <p><strong>Energy:</strong> {lattice.ms_predictions.energy.toFixed(4)}</p>
                            )}
                            {/* Add more fields as needed */}
                        </div>
                    ))}
                </div>
                {/* Load More Button */}
                {nextPageLastId && (
                     <div className="mt-6 text-center">
                        <button
                            onClick={(e) => handleSubmit(e, true)} // Pass true for loadMore
                            className={`px-4 py-2 text-white font-semibold rounded-md transition duration-200 ease-in-out text-sm ${
                                isLoading
                                    ? 'bg-gray-400 cursor-not-allowed'
                                    : 'bg-emerald-600 hover:bg-emerald-700'
                            }`}
                            disabled={isLoading}
                        >
                            {isLoading ? 'Loading...' : 'Load More'}
                        </button>
                    </div>
                )}
            </div>


        </div> // Closing main container div
    );
}

export default RetrieveLattice;