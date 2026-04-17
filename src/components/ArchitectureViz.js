"use client";
import React from 'react';

// ── Layer definitions per model ───────────────────────────────────────────────
const ARCHITECTURES = {
  cnn: {
    title: '1-D CNN Architecture',
    subtitle: 'Filters slide along the time axis to detect local patterns',
    layers: [
      { name: 'Input',          shape: '[N × 16 × 1]',  detail: '16 timesteps, 1 feature',        color: '#475569', icon: '📥' },
      { name: 'Conv1D',         shape: '[N × 16 × 16]', detail: 'filters=16, kernel=3, ReLU, same', color: '#2563eb', icon: '🔍' },
      { name: 'MaxPool1D',      shape: '[N × 8 × 16]',  detail: 'pool_size=2 → halve sequence',   color: '#7c3aed', icon: '⬇' },
      { name: 'Conv1D',         shape: '[N × 8 × 32]',  detail: 'filters=32, kernel=3, ReLU, same', color: '#2563eb', icon: '🔍' },
      { name: 'GlobalMaxPool',  shape: '[N × 32]',       detail: 'take max over time axis',        color: '#7c3aed', icon: '⬇' },
      { name: 'Dense',          shape: '[N × 32]',       detail: 'ReLU activation',                color: '#0d9488', icon: '○' },
      { name: 'Output',         shape: '[N × 1]',        detail: 'Sigmoid → binary class prob',    color: '#16a34a', icon: '✓' },
    ],
  },
  rnn: {
    title: 'RNN Architecture',
    subtitle: 'Hidden state carries information from each step to the next',
    layers: [
      { name: 'Input',      shape: '[N × 16 × 1]',  detail: '16 timesteps, 1 feature',   color: '#475569', icon: '📥' },
      { name: 'SimpleRNN',  shape: '[N × 32]',       detail: 'units=32, hₜ = tanh(Wxₜ + Uhₜ₋₁)', color: '#b45309', icon: '↻' },
      { name: 'Dense',      shape: '[N × 16]',       detail: 'ReLU activation',            color: '#0d9488', icon: '○' },
      { name: 'Output',     shape: '[N × 1]',        detail: 'Sigmoid → binary class prob', color: '#16a34a', icon: '✓' },
    ],
  },
  lstm: {
    title: 'LSTM Architecture',
    subtitle: 'Gates control what to remember, forget, and output at each step',
    layers: [
      { name: 'Input',  shape: '[N × 16 × 1]',  detail: '16 timesteps, 1 feature',          color: '#475569', icon: '📥' },
      { name: 'LSTM',   shape: '[N × 32]',       detail: 'units=32 — forget / input / output gates', color: '#7c3aed', icon: '⊞' },
      { name: 'Dense',  shape: '[N × 16]',       detail: 'ReLU activation',                  color: '#0d9488', icon: '○' },
      { name: 'Output', shape: '[N × 1]',        detail: 'Sigmoid → binary class prob',      color: '#16a34a', icon: '✓' },
    ],
  },
  transformer: {
    title: 'Transformer Architecture',
    subtitle: 'Self-attention lets every timestep attend to every other timestep',
    layers: [
      { name: 'Input',           shape: '[N × 16 × 1]',  detail: '16 timesteps, 1 feature',              color: '#475569', icon: '📥' },
      { name: 'Linear Embed',    shape: '[N × 16 × 8]',  detail: 'project 1 → 8 dims per token',         color: '#0e7490', icon: '→' },
      { name: 'Self-Attention',  shape: '[N × 16 × 8]',  detail: 'Q·Kᵀ/√d softmax → weighted V, d=8',  color: '#9333ea', icon: '⚡' },
      { name: 'GlobalAvgPool',   shape: '[N × 8]',        detail: 'average over 16 token vectors',        color: '#7c3aed', icon: '⬇' },
      { name: 'Dense',           shape: '[N × 16]',       detail: 'ReLU activation',                      color: '#0d9488', icon: '○' },
      { name: 'Output',          shape: '[N × 1]',        detail: 'Sigmoid → binary class prob',          color: '#16a34a', icon: '✓' },
    ],
  },
};

// ── Single layer card ────────────────────────────────────────────────────────
function LayerCard({ layer, isLast }) {
  return (
    <div className="flex flex-col items-center">
      <div className="w-full rounded-lg px-3 py-2 flex items-start gap-2"
        style={{ background: `${layer.color}22`, border: `1px solid ${layer.color}55` }}>
        <span className="text-base leading-none mt-0.5 shrink-0">{layer.icon}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2 flex-wrap">
            <span className="text-xs font-bold text-white">{layer.name}</span>
            <span className="font-mono text-[9px] text-slate-400">{layer.shape}</span>
          </div>
          <div className="text-[9px] text-slate-500 mt-0.5 leading-tight">{layer.detail}</div>
        </div>
      </div>
      {!isLast && (
        <div className="text-slate-600 text-sm leading-none my-0.5">↓</div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
export default function ArchitectureViz({ algo }) {
  const arch = ARCHITECTURES[algo];
  if (!arch) return null;

  return (
    <div className="flex flex-col h-full overflow-y-auto bg-slate-950/40 rounded-xl p-3 gap-2">
      <div className="shrink-0">
        <div className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">{arch.title}</div>
        <div className="text-[10px] text-slate-500 mt-0.5 leading-snug">{arch.subtitle}</div>
      </div>
      <div className="flex flex-col gap-0">
        {arch.layers.map((layer, i) => (
          <LayerCard key={i} layer={layer} isLast={i === arch.layers.length - 1} />
        ))}
      </div>
    </div>
  );
}
