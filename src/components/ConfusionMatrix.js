"use client";
import React from 'react';

/**
 * Renders a color-coded confusion matrix.
 * Props:
 *   matrix  – number[][]  (actual × predicted)
 *   labels  – string[]    class labels in the same order
 */
const ConfusionMatrix = ({ matrix, labels }) => {
  if (!matrix || !labels || matrix.length === 0) return null;

  const n   = labels.length;
  const max = Math.max(...matrix.flat(), 1);

  // Diagonal = correct (green), off-diagonal = errors (red)
  const cellBg = (val, i, j) => {
    const intensity = val / max;
    if (i === j)
      return `rgba(16, 185, 129, ${0.12 + intensity * 0.72})`; // green
    return val === 0 ? 'transparent' : `rgba(239, 68, 68, ${intensity * 0.55})`;
  };

  const cellText = (val, intensity) =>
    intensity > 0.25 ? '#e2e8f0' : val === 0 ? '#334155' : '#94a3b8';

  // Sum per actual row → per-class recall
  const rowSums = matrix.map(row => row.reduce((a, b) => a + b, 0));
  const total   = rowSums.reduce((a, b) => a + b, 0);
  const correct = matrix.reduce((s, row, i) => s + row[i], 0);
  const accuracy = total > 0 ? ((correct / total) * 100).toFixed(1) : '—';

  return (
    <div className="bg-slate-900/80 rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">
          Confusion Matrix
        </h3>
        <span className="text-[10px] font-mono text-emerald-400">{accuracy}% acc</span>
      </div>

      <div className="overflow-x-auto">
        <table className="mx-auto border-collapse text-[10px] font-mono">
          <thead>
            <tr>
              {/* top-left empty corner */}
              <th className="w-6 pb-1" />
              <th className="pb-1 text-slate-500" colSpan={n} style={{ fontSize: 9 }}>
                Predicted →
              </th>
            </tr>
            <tr>
              <th />
              {labels.map(l => (
                <th key={l} className="w-8 h-6 text-center text-slate-400 font-medium"
                  style={{ fontSize: 10 }}>
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                {/* row label */}
                {i === 0 && (
                  <td rowSpan={n} className="pr-1 text-slate-500 align-middle"
                    style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', fontSize: 9, textAlign: 'center' }}>
                    Actual ↑
                  </td>
                )}
                {row.map((val, j) => {
                  const intensity = val / max;
                  return (
                    <td key={j}
                      className="w-8 h-8 text-center rounded transition-all text-[11px] font-bold"
                      style={{
                        background: cellBg(val, i, j),
                        color: cellText(val, intensity),
                        border: i === j ? '1px solid rgba(16,185,129,0.3)' : '1px solid transparent',
                      }}>
                      {val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex gap-3 justify-center text-[9px] text-slate-500">
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/50 inline-block" /> Correct
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm bg-red-500/40 inline-block" /> Incorrect
        </span>
      </div>
    </div>
  );
};

export default ConfusionMatrix;
