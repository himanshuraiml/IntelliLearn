"use client";
import React, { useState } from 'react';

const ParameterControl = ({ label, value, min, max, step, onChange, hint, warning }) => {
  const [showHint, setShowHint] = useState(false);
  const warningText = warning ? warning(value) : null;

  return (
    <div className="flex flex-col space-y-1.5 mb-3">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-1">
          <label className="text-xs font-medium text-slate-400">{label}</label>
          {hint && (
            <button
              onClick={() => setShowHint(v => !v)}
              className="w-4 h-4 rounded-full bg-slate-700 text-slate-500 hover:text-slate-300 flex items-center justify-center text-[9px] font-bold leading-none transition-colors"
              title="Show hint"
            >
              ?
            </button>
          )}
        </div>
        <span className="text-xs font-mono bg-brand-500/10 text-brand-500 px-2 py-0.5 rounded">{value}</span>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-brand-500"
      />

      {showHint && hint && (
        <p className="text-[10px] text-slate-500 leading-snug bg-slate-800/60 rounded px-2 py-1 border border-white/5">
          {hint}
        </p>
      )}

      {warningText && (
        <p className="text-[10px] text-amber-400/80 leading-snug flex items-center gap-1">
          <span>⚠</span> {warningText}
        </p>
      )}
    </div>
  );
};

export default ParameterControl;
