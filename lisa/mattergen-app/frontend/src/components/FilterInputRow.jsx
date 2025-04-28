function FilterInputRow({ fieldConfig, filterState, onChange, isLoading }) {
    const { name, label, numberType, placeholder, min } = fieldConfig;
    const { value, op } = filterState;

    const opBackendMapping = {
        '=': 'eq',
        '!=': 'neq',
        '<': 'lt',
        '<=': 'lte',
        '>': 'gt',
        '>=': 'gte'
    };

    const opOptions = Object.keys(opBackendMapping);
    
    // Don't show operator for Limit field
    const showOperator = name !== 'Limit';

    // Map label to display name (capitalize and insert spaces)
    const displayLabel = label.charAt(0).toUpperCase() + label.slice(1).replace(/([A-Z])/g, ' $1').trim();

    return (
        <div>
            <label htmlFor={`${name}-value`} className="block text-md font-semibold text-emerald-700 mb-1">
                {displayLabel} {name == 'Limit' && <span className="text-red-500">*</span>}
            </label>
            <p className="text-xs text-gray-500 mb-1">
                {name === 'Limit' ? 'Maximum number of results to display per page.' : `Provide a ${numberType} value for filtering.`}
            </p>
            <div className={`mt-1 flex space-x-2 ${!showOperator ? 'w-full' : ''}`}>
                {/* Operator Dropdown (conditional) */}
                {showOperator && (
                    <select
                        id={`${name}-op`}
                        name={`${name}-op`}
                        value={op}
                        onChange={(e) => onChange(name, 'op', e.target.value)}
                        className="w-1/4 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500 text-sm"
                        disabled={isLoading}
                    >
                        {opOptions.map(option => (
                            // <option key="=" value="=">=</option>
                            // value is what's sent to backend
                            <option key={option} value={option}>
                                {option}
                            </option>
                        ))}
                    </select>
                )}
                {/* Value Input */}
                <input
                    type={"number"}
                    id={`${name}-value`}
                    name={`${name}-value`}
                    value={value}
                    step={numberType === 'integer' ? '1' : 'any'}
                    onChange={(e) => onChange(name, 'value', e.target.value)}
                    placeholder={placeholder}
                    min={min}
                    className={`text-sm flex-grow px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-emerald-500 focus:border-emerald-500 ${!showOperator ? 'w-full' : 'w-3/4'}`}
                    disabled={isLoading}
                />
            </div>
        </div>
    );
}

export default FilterInputRow;