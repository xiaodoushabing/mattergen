import React, { useState } from 'react';
import RetrieveLatticeByFilter from './RetrieveLatticeByFilter';
import RetrieveLatticeById from './RetrieveLatticeById';

function RetrieveLattice() {
    const [retrievalMethod, setRetrievalMethod] = useState("filter"); // Default to 'filter'

    console.log("Current retrievalMethod:", retrievalMethod);
    
    return (
        <>
            {/* Container 1: Heading and Toggle Buttons */}
            <div className="relative z-10 p-7 bg-slate-50 rounded-2xl shadow-lg border border-stone-200 max-w-2xl w-full mx-auto mb-8">
                <h2 className=" relative z-10 text-2xl font-semibold text-slate-800 text-center">
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
                        {' or '}
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
                <RetrieveLatticeById />
            )}
        </>
    );
}

export default RetrieveLattice;