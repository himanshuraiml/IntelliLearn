"use client";
import React from 'react';

const ParameterControl = ({ label, value, min, max, step, onChange }) => {
  return (
    <div className="flex flex-col space-y-2 mb-4">
      <div className="flex justify-between items-center">
        <label className="text-sm font-medium text-slate-400">{label}</label>
        <span className="text-xs font-mono bg-brand-500/10 text-brand-500 px-2 py-0.5 rounded">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-brand-500"
      />
    </div>
  );
};

export default ParameterControl;
