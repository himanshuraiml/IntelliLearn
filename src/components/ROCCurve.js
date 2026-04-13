"use client";
import React, { useMemo } from 'react';

/**
 * Computes ROC curve points + AUC using raw probability scores.
 * getRawProb(x, y) → number in [0, 1] (P(class=1))
 */
function computeROC(data, getRawProb) {
  const binaryData = data.filter(d => d.label === 0 || d.label === 1);
  if (binaryData.length < 4) return null;

  const scores = binaryData
    .map(d => ({ prob: getRawProb(d.x, d.y), label: d.label }))
    .sort((a, b) => b.prob - a.prob);

  const nPos = scores.filter(s => s.label === 1).length;
  const nNeg = scores.filter(s => s.label === 0).length;
  if (nPos === 0 || nNeg === 0) return null;

  const points = [{ fpr: 0, tpr: 0 }];
  let tp = 0, fp = 0;
  scores.forEach(({ label }) => {
    if (label === 1) tp++;
    else fp++;
    points.push({ fpr: fp / nNeg, tpr: tp / nPos });
  });
  points.push({ fpr: 1, tpr: 1 });

  // AUC via trapezoidal rule
  let auc = 0;
  for (let i = 1; i < points.length; i++) {
    auc += (points[i].fpr - points[i - 1].fpr) * (points[i].tpr + points[i - 1].tpr) / 2;
  }

  return { points, auc: Math.max(0, Math.min(1, auc)) };
}

const ROCCurve = ({ data, getRawProb }) => {
  const roc = useMemo(() => {
    if (!getRawProb || !data?.length) return null;
    return computeROC(data, getRawProb);
  }, [data, getRawProb]);

  if (!roc) return null;

  const { points, auc } = roc;
  const W   = 220;
  const H   = 175;
  const pad = { top: 12, right: 12, bottom: 28, left: 30 };
  const iW  = W - pad.left - pad.right;
  const iH  = H - pad.top  - pad.bottom;

  const sx  = fpr => pad.left + fpr * iW;
  const sy  = tpr => pad.top  + (1 - tpr) * iH;

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.fpr).toFixed(1)},${sy(p.tpr).toFixed(1)}`).join(' ');
  const fillD = pathD + ` L${sx(1).toFixed(1)},${sy(0).toFixed(1)} L${sx(0).toFixed(1)},${sy(0).toFixed(1)} Z`;
  const diagD = `M${sx(0)},${sy(0)} L${sx(1)},${sy(1)}`;

  const aucColor = auc >= 0.9 ? '#22c55e' : auc >= 0.7 ? '#f59e0b' : '#ef4444';
  const aucLabel = auc >= 0.9 ? 'Excellent' : auc >= 0.8 ? 'Good' : auc >= 0.7 ? 'Fair' : 'Poor';

  return (
    <div className="bg-slate-900/80 rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">ROC Curve</h3>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-slate-500">{aucLabel}</span>
          <span
            className="text-xs font-mono font-bold px-2 py-0.5 rounded-md"
            style={{ background: aucColor + '18', color: aucColor }}
          >
            AUC {auc.toFixed(3)}
          </span>
        </div>
      </div>

      <svg width={W} height={H} className="max-w-full block">
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map(v => (
          <g key={v}>
            <line x1={sx(v)} y1={pad.top} x2={sx(v)} y2={pad.top + iH} stroke="#1e293b" strokeWidth={1} />
            <line x1={pad.left} y1={sy(v)} x2={pad.left + iW} y2={sy(v)} stroke="#1e293b" strokeWidth={1} />
          </g>
        ))}

        {/* Diagonal reference (random classifier) */}
        <path d={diagD} fill="none" stroke="#334155" strokeWidth={1} strokeDasharray="5,3" />

        {/* AUC fill */}
        <path d={fillD} fill="#a78bfa" opacity={0.09} />

        {/* ROC curve */}
        <path d={pathD} fill="none" stroke="#a78bfa" strokeWidth={2.5} strokeLinejoin="round" strokeLinecap="round" />

        {/* Axes */}
        <line x1={pad.left} y1={pad.top + iH} x2={pad.left + iW} y2={pad.top + iH} stroke="#334155" strokeWidth={1} />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + iH} stroke="#334155" strokeWidth={1} />

        {/* Axis labels */}
        <text x={pad.left + iW / 2} y={H - 4} textAnchor="middle" fill="#64748b" fontSize={9}>
          False Positive Rate
        </text>
        <text
          x={9}
          y={pad.top + iH / 2}
          textAnchor="middle"
          fill="#64748b"
          fontSize={9}
          transform={`rotate(-90, 9, ${pad.top + iH / 2})`}
        >
          True Positive Rate
        </text>

        {/* Tick labels */}
        {[0, 0.5, 1].map(v => (
          <g key={v}>
            <text x={sx(v)} y={pad.top + iH + 10} textAnchor="middle" fill="#475569" fontSize={8}>{v}</text>
            <text x={pad.left - 4} y={sy(v) + 3}   textAnchor="end"    fill="#475569" fontSize={8}>{v}</text>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default ROCCurve;
