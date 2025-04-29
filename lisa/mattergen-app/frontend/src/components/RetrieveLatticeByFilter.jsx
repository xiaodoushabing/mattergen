import React, { useState } from "react";
import axios from "axios";
import FilterInputRow from "./FilterInputRow"
import { parseAtomsList } from "../utils/parsers";

// const API = import.meta.env.VITE_API_URL;
const API = "http://localhost:8000"

function RetrieveLatticeByFilter () {
    // Define configuration for each filter field
    const filterFieldsConfig = [
    { name: 'Limit', label: 'limit', defaultOp: '=', defaultValue: '10', placeholder: 'e.g., 10', numberType: 'integer', min: 1 },
    { name: 'LatticeIndex', label: 'latticeIndex', defaultOp: '=', defaultValue: '', placeholder: 'e.g., 5', numberType: 'integer', min: 1 },
    { name: 'GuidanceFactor', label: 'guidanceFactor', defaultOp: '=', defaultValue: '', placeholder: 'e.g., 4.0', numberType: 'float', min: 1 },
    { name: 'MagneticDensity', label: 'magneticDensity', defaultOp: '=', defaultValue: '', placeholder: 'e.g., 0.5', numberType: 'float', min: 0 },
    { name: 'NoOfAtoms', label: 'numberOfAtoms', defaultOp: '=', defaultValue: '', placeholder: 'e.g., 12', numberType: 'integer', min: 1 },
    { name: 'Energy', label: 'energy', defaultOp: '=', defaultValue: '', placeholder: 'e.g., -330.9', numberType: 'float' },
];

    const opBackendMapping = {
            '=': 'eq',
            '!=': 'neq',
            '<': 'lt',
            '<=': 'lte',
            '>': 'gt',
            '>=': 'gte'
        };

    const opOptions = Object.keys(opBackendMapping);

    const [filters, setFilters] = useState(() => {
        const initialState = {};
        filterFieldsConfig.forEach(field => {
            initialState[field.name] = {
                op: field.defaultOp,
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
        
        const { data: atoms_list, error: atomsError } = parseAtomsList(filters.AtomsList);

        if (atomsError) {
            setError(atomsError);
            setIsLoading(false);
            return;
        }
        
        if (atoms_list && atoms_list.length > 0) {
            requestPayload.atoms_list = atoms_list;
        }

        filterFieldsConfig.forEach(field => {
            const filterState = filters[field.name];
            const filterValue = filterState.value;
            const filterOp = filterState.op;
            const filterType = field.numberType;

            if (!opOptions.includes(filterOp)) {
                const invalidLabel = field.label.replace(/([A-Z])/g, ' $1').trim();
                setError(`Invalid operator "${filterOp}" selected for ${invalidLabel}. Please choose a valid operator from the list.`);
                setIsLoading(false);
                return;
            }

            if (filterValue) {
                const valueNum = (filterType === 'float') ? parseFloat(filterValue) : parseInt(filterValue);

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

                if (field.name === 'Limit') {
                    requestPayload.limit = valueNum;
                    return;
                }
                // map state key to backend snakecase format
                const backendOpCode = opBackendMapping[filterOp];
                if (!backendOpCode) { // Should not happen if operator validation passed
                     validationError = `Internal error mapping operator "${filterOp}".`; return;
                }

                const apiFieldName = field.name.replace(/([A-Z])/g, '_$1').toLowerCase().replace(/^_/, '');
                requestPayload[apiFieldName] = { value: valueNum, op: backendOpCode };
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

    const InfoField = ({ label, value }) => (
        <p className="mb-1">
            <strong>{label}</strong>
            <span className="block ml-4 text-gray-700">{value}</span>
        </p>
    );

    return (
        <div className="relative z-10 p-8 bg-slate-50 rounded-2xl shadow-[0_4px_30px_rgba(0,0,0,0.1)] border border-stone-200 max-w-3xl w-full mx-auto">
        {/* <h2 className="relative z-10 text-3xl font-semibold text-slate-800 mb-10 mt-5 text-center">
            Retrieve Lattices by Filters
        </h2> */}

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
                    <label htmlFor={`atomsList`} className="block text-md font-semibold text-emerald-700 mb-1">
                        Atoms List
                    </label>
                    <p className="text-xs text-gray-500 mb-1">
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
                <div className="flex justify-center"> {/* Added padding top */}
                    <button
                        type="submit"
                        className={`bg-gradient-to-r from-emerald-500 to-teal-400 hover:from-emerald-700 hover:to-teal-600 text-stone-100 font-bold text-lg py-3 px-6 rounded-2xl shadow-lg hover:shadow-xl transition duration-300 ease-in-out ${
                        isLoading
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-teal-800 hover:bg-emerald-700'
                    }`}
                    disabled={isLoading}
                >
                    {isLoading ? 'Retrieving lattices...' : 'Retrieve Lattices'}
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
            <div className="mt-8">
                {results.length > 0 && (
                    <h3 className="text-2xl font-bold text-slate-700 mb-6">
                        Results:
                    </h3>
                )}
                <div className="space-y-4">
                    {results.map((lattice) => (
                        <div
                            key={lattice._id}
                            className="p-4 border border-gray-200 rounded-lg bg-gray-100 flex justify-between items-start text-sm"
                        >
                            {/* Main Content Area (takes available space on the left) */}
                            <div>
                                <InfoField label="Number of atoms:" value={lattice.no_of_atoms} />
                                <InfoField label="Elements:" value=
                                    {Object.entries(lattice.atoms_list) // Get [key, value] pairs
                                        .map(([element, count], index, arr) => (
                                            <span key={element}>
                                                <span className="font-semibold text-emerald-700">{element}</span>
                                                :{' '}
                                                <span className="text-gray-600">{count}</span>
                                                {index < arr.length - 1 ? ', ' : ''}
                                            </span>
                                        ))
                                    } />
                                
                                <InfoField label="Guidance Factor:" value={lattice.guidance_factor} />
                                <InfoField label="Magnetic Density:" value={lattice.magnetic_density} />
                                 
                                {lattice.ms_predictions?.energy && (
                                    <InfoField label="Energy:" value={lattice.ms_predictions.energy.toFixed(2)} />
                                )}
                            </div>

                            {/* ID and Index Area (pushed to the top right) */}
                            <div className="text-right text-xs text-gray-500 ml-4">
                                <p className="font-semibold mb-1">
                                    ID: {lattice._id}
                                </p>
                                <p>
                                    Index: {lattice.lattice_index}
                                </p>
                            </div>
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
                            {isLoading ? 'Fetching...' : 'Load More'}
                        </button>
                    </div>
                )}
            </div>


        </div>
    );
}

export default RetrieveLatticeByFilter;