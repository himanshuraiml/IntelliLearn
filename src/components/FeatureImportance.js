"use client";
import React from 'react';

/**
 * Horizontal bar chart showing feature importances.
 * Props:
 *   importance – { x: number, y: number }  (values 0–1, should sum ≈ 1)
 *   modelType  – string (e.g. 'decisionTree', 'randomForest')
 */
const FeatureImportance = ({ importance, modelType }) => {
  if (!importance) return null;

  const features = [
    { name: 'Feature X', key: 'x', value: importance.x ?? 0 },
    { name: 'Feature Y', key: 'y', value: importance.y ?? 0 },
  ];

  // Normalise so bars always span 0–100 %
  const total = features.reduce((s, f) => s + f.value, 0) || 1;
  const bars  = features.map(f => ({ ...f, pct: (f.value / total) * 100 }));
  bars.sort((a, b) => b.pct - a.pct);

  const modelLabel = modelType === 'randomForest' ? 'Random Forest' : 'Decision Tree';

  const barColors = ['#6366f1', '#10b981'];

  return (
    <div className="bg-slate-900/80 rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">
          Feature Importance
        </h3>
        <span className="text-[10px] text-slate-500">{modelLabel}</span>
      </div>

      <p className="text-[10px] text-slate-600 mb-3 leading-snug">
        How much each input contributes to splitting decisions (Gini gain).
      </p>

      <div className="space-y-3">
        {bars.map((f, i) => (
          <div key={f.key}>
            <div className="flex justify-between text-[10px] mb-1">
              <span className="text-slate-400 font-medium">{f.name}</span>
              <span className="font-mono text-slate-500">{f.pct.toFixed(1)}%</span>
            </div>
            <div className="w-full h-4 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${f.pct}%`,
                  background: barColors[i % barColors.length],
                  opacity: 0.85,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <p className="text-[9px] text-slate-600 mt-3 leading-snug">
        Higher importance → the model relies more on this feature to make decisions.
        Equal importances suggest both features are equally useful.
      </p>
    </div>
  );
};

export default FeatureImportance;
