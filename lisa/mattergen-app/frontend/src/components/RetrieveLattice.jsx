// FILE: frontend/src/components/RetrieveLattice.jsx
import React, { useState } from 'react';
// Assuming RetrieveLatticeByFilter is imported or defined
import RetrieveLatticeByFilter from './RetrieveLatticeByFilter';

function RetrieveLattice() {
    const [retrievalMethod, setRetrievalMethod] = useState("filter"); // Default to 'filter'

    return (
        <>
            {/* Container 1: Heading and Toggle Buttons */}
            <div className="relative z-10 p-5 bg-cyan-50 rounded-2xl shadow-lg border border-stone-200 max-w-lg w-full mx-auto mb-8">
                <h2 className="text-2xl font-semibold text-slate-800 text-center">
                    Retrieve Lattices by {' '}
                    <div className="inline-block ml-2">
                        <button
                            onClick={() => setRetrievalMethod("filter")}
                            className={`font-semibold py-1 px-3 border rounded shadow transition duration-150 ease-in-out ${
                                retrievalMethod === 'filter'
                                    ? 'bg-emerald-600 text-white border-emerald-700'
                                    : 'bg-slate-200 hover:bg-slate-300 text-slate-800 border-slate-300'
                            }`}
                        >
                            Filter
                        </button>
                        {' '}
                        <button
                            onClick={() => setRetrievalMethod("id")}
                            className={`font-semibold py-1 px-3 border rounded shadow transition duration-150 ease-in-out ${
                                retrievalMethod === 'id'
                                    ? 'bg-emerald-600 text-white border-emerald-700'
                                    : 'bg-slate-200 hover:bg-slate-300 text-slate-800 border-slate-300'
                            }`}
                        >
                            ID
                        </button>
                    </div>
                </h2>
            </div>

            {retrievalMethod === "filter" ? (
                <RetrieveLatticeByFilter />
            ) : (
                <div className="text-center w-full max-w-4xl mx-auto p-8 bg-white rounded-2xl shadow-lg border border-stone-200"> {/* Added container styling */}
                    <h3 className="text-xl font-semibold text-slate-700 mb-4">
                        Retrieve Lattice by ID
                    </h3>
                    <p className="text-md text-slate-500">
                        This feature is not yet implemented.
                    </p>
                </div>
            )}
        </>
    );
}

export default RetrieveLattice;