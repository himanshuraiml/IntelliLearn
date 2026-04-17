"use client";
import React, { useMemo } from 'react';

// ── Mini sparkline for one sequence ──────────────────────────────────────────
function Sparkline({ seq, label, prediction, width = 100, height = 44 }) {
  if (!seq || seq.length === 0) return null;
  const pad = 4;
  const w = width  - pad * 2;
  const h = height - pad * 2;
  const n = seq.length;

  const points = seq.map((v, i) => {
    const x = pad + (i / (n - 1)) * w;
    const y = pad + (1 - v) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const correct   = prediction == null ? null : prediction.correct;
  const borderCol = prediction == null
    ? (label === 1 ? '#3b82f6' : '#f97316')
    : (correct ? '#22c55e' : '#ef4444');
  const lineCol   = label === 1 ? '#60a5fa' : '#fb923c';

  return (
    <svg width={width} height={height}
      style={{ border: `1.5px solid ${borderCol}`, borderRadius: 6, background: '#0f172a' }}>
      <polyline points={points} fill="none" stroke={lineCol} strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round" />
      {prediction != null && !correct && (
        <text x={width - 5} y={10} textAnchor="end" fontSize="8" fill="#ef4444">✗</text>
      )}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
export default function SequencePlot({ data, predictions, isTraining }) {
  // Separate samples by class — only process valid sequence data
  const byClass = useMemo(() => {
    const c0 = [], c1 = [];
    data.forEach((d, i) => {
      if (!d.seq) return; // skip old-format {x, y, label} points during transition
      const entry = { seq: d.seq, label: d.label, idx: i };
      if (d.label === 0) c0.push(entry); else c1.push(entry);
    });
    return { 0: c0.slice(0, 8), 1: c1.slice(0, 8) };
  }, [data]);

  const accuracy = predictions
    ? (predictions.filter(p => p.correct).length / predictions.length * 100).toFixed(1)
    : null;

  return (
    <div className="flex flex-col h-full overflow-hidden bg-slate-950/40 rounded-xl p-3 gap-3">

      {/* Header */}
      <div className="flex items-center justify-between shrink-0">
        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">
          Sample Sequences
        </span>
        {accuracy != null && (
          <span className="text-xs font-mono font-bold text-green-400">
            {accuracy}% accurate
          </span>
        )}
        {isTraining && (
          <span className="text-xs text-brand-400 animate-pulse">Training…</span>
        )}
      </div>

      {/* Class columns */}
      <div className="flex gap-3 flex-1 overflow-hidden min-h-0">
        {[0, 1].map(cls => (
          <div key={cls} className="flex-1 flex flex-col gap-1 min-w-0">
            <div className="text-[10px] font-semibold mb-1 shrink-0"
              style={{ color: cls === 1 ? '#60a5fa' : '#fb923c' }}>
              Class {cls}
            </div>
            <div className="flex flex-wrap gap-1 content-start overflow-hidden">
              {byClass[cls].map(({ seq, label, idx }) => (
                <Sparkline
                  key={idx}
                  seq={seq}
                  label={label}
                  prediction={predictions ? predictions[idx] : null}
                  width={90}
                  height={42}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="shrink-0 text-[9px] text-slate-600 space-y-0.5">
        <div className="flex gap-3">
          <span><span style={{ color: '#22c55e' }}>■</span> correct prediction</span>
          <span><span style={{ color: '#ef4444' }}>■</span> wrong prediction</span>
        </div>
        <div>Each sparkline = 1 sequence of {data.find(d => d.seq)?.seq?.length ?? 16} timesteps</div>
      </div>
    </div>
  );
}
