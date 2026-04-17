"use client";
import React, { useState, useEffect } from 'react';
import { X, ChevronLeft, ChevronRight, Brain } from 'lucide-react';

// ─── Concrete example numbers used throughout ────────────────────────────────
const E = {
  x1: 0.5,  x2: 0.3,
  w1: 0.80, w2: -0.40, b: 0.10,
  z: 0.38,  a: 0.594,  y: 1,  loss: 0.521,
  delta: -0.406, sp: 0.241,
  gw1: -0.203, gw2: -0.122, gb: -0.406,
  w1n: 0.802,  w2n: -0.399, bn: 0.104,
  lr: 0.01,
};

// ─── Step definitions ────────────────────────────────────────────────────────
const STEPS = [
  {
    title: 'What is Backpropagation?',
    subtitle: 'The algorithm that trains every neural network',
    body: [
      'A neural network has hundreds of weights W and biases b. During training we need to answer one question for every weight: "how much does changing this weight increase or decrease the loss?"',
      'Backpropagation computes ∂L/∂w — the gradient of the loss — for every weight in a single backward pass. Without it, training deep networks would be computationally impossible.',
    ],
    math: [
      { lbl: 'Prediction', eq: 'ŷ = forward_pass(x, W, b)' },
      { lbl: 'Loss (BCE)', eq: 'L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]' },
      { lbl: 'Goal',       eq: 'Find  ∂L/∂w  for ALL weights w' },
    ],
    note: '"Backprop is just the chain rule, applied cleverly across layers."',
    hl: 'intro',
  },
  {
    title: 'Step 1 — Forward Pass: Weighted Sum z',
    subtitle: 'Each neuron computes a linear combination of its inputs',
    body: [
      'Before activation, each neuron computes z = Wx + b — the dot product of the weight vector and input vector, plus a bias. This pre-activation is the "raw signal" entering the non-linearity.',
    ],
    math: [
      { lbl: 'Formula', eq: 'z  =  w₁·x₁  +  w₂·x₂  +  b' },
      { lbl: 'Values',  eq: `z  =  ${E.w1}×${E.x1}  +  (${E.w2})×${E.x2}  +  ${E.b}` },
      { lbl: '     ',   eq: '   =  0.40  −  0.12  +  0.10' },
      { lbl: 'Result',  eq: `z  =  ${E.z}` },
    ],
    note: 'z can be any real number. The activation function will squash it into a useful range next.',
    hl: 'fwd-z',
  },
  {
    title: 'Step 2 — Forward Pass: Activation a = σ(z)',
    subtitle: 'Non-linearity is what makes deep networks powerful',
    body: [
      'The activation function introduces non-linearity. Without it, stacking layers would still produce just a linear model. Sigmoid squashes any z into (0, 1) — perfect for binary classification probabilities.',
      'Its derivative has a beautiful closed form: σ\'(z) = a(1−a). This will appear in every backprop calculation.',
    ],
    math: [
      { lbl: 'Sigmoid',    eq: 'σ(z)  =  1 / (1 + e^−z)' },
      { lbl: 'Numbers',    eq: `a  =  σ(${E.z})  ≈  ${E.a}` },
      { lbl: 'Derivative', eq: "σ'(z)  =  σ(z) · (1 − σ(z))  =  a(1−a)" },
      { lbl: 'Example',    eq: `σ'(${E.z})  =  ${E.a} × ${(1-E.a).toFixed(3)}  ≈  ${E.sp}` },
    ],
    note: 'σ\'(z) ≤ 0.25 always. This "gates" how much gradient can pass through — key to vanishing gradients.',
    hl: 'fwd-a',
  },
  {
    title: 'Step 3 — Computing the Loss L',
    subtitle: 'Measure how wrong the prediction is',
    body: [
      'After the full forward pass produces ŷ (our prediction), we compute how wrong it is using Binary Cross-Entropy. The loss is large when we are confidently wrong and near zero when we are confidently correct.',
    ],
    math: [
      { lbl: 'BCE',     eq: 'L  =  −[y·log(ŷ) + (1−y)·log(1−ŷ)]' },
      { lbl: 'y=1 →',  eq: 'L  =  −log(ŷ)' },
      { lbl: 'Numbers', eq: `L  =  −log(${E.a})  ≈  ${E.loss}` },
      { lbl: 'Perfect', eq: 'ŷ → 1.0  ⟹  L → 0  ✓' },
    ],
    note: 'Loss 0.521 means we\'re uncertain. The model needs to push ŷ closer to 1.',
    hl: 'loss',
  },
  {
    title: 'Chain Rule — Backprop\'s Engine',
    subtitle: 'Decompose ∂L/∂w into a product of local gradients',
    body: [
      'The chain rule from calculus: if L depends on a, and a depends on z, and z depends on w, then ∂L/∂w is the product of three local derivatives. Each piece is trivial to compute.',
      'Backprop reuses intermediate δ values layer by layer — computing ALL gradients in one pass.',
    ],
    math: [
      { lbl: 'Chain Rule', eq: '∂L/∂w  =  ∂L/∂a · ∂a/∂z · ∂z/∂w' },
      { lbl: '           ', eq: '          ↑         ↑         ↑' },
      { lbl: '           ', eq: '       loss→a    a→z      z→w' },
      { lbl: 'Multi-layer', eq: "δ^(l) = (W^(l+1))ᵀ · δ^(l+1) ⊙ σ'(z^(l))" },
    ],
    note: 'Backprop computes N gradients in O(N) time. Finite differences would need O(N²)!',
    hl: 'chain',
  },
  {
    title: 'Step 4 — Output Gradient δ = ŷ − y',
    subtitle: 'The backward pass starts at the loss',
    body: [
      'We begin the backward pass at the output. For sigmoid + Binary Cross-Entropy, ∂L/∂z simplifies beautifully to ŷ − y. This elegant result is the "error signal" δ that flows backward.',
    ],
    math: [
      { lbl: '∂L/∂ŷ',    eq: '= −y/ŷ  +  (1−y)/(1−ŷ)' },
      { lbl: '∂ŷ/∂z',    eq: "= σ'(z)  =  ŷ(1−ŷ)" },
      { lbl: 'Combined δ', eq: 'δ  =  ∂L/∂z  =  ŷ − y   (BCE+sigmoid)' },
      { lbl: 'Numbers',   eq: `δ  =  ${E.a} − ${E.y}  =  ${E.delta}` },
    ],
    note: 'δ = −0.406: negative means output was too low. Backprop will push ŷ higher.',
    hl: 'bwd-output',
  },
  {
    title: 'Step 5 — Weight Gradient ∂L/∂w',
    subtitle: 'How does each weight affect the pre-activation z?',
    body: [
      'Since z = w₁x₁ + w₂x₂ + b, the gradient ∂z/∂wᵢ is simply xᵢ. Combined with δ via the chain rule, the final weight gradient is δ·xᵢ — beautifully simple.',
    ],
    math: [
      { lbl: '∂z/∂w₁',  eq: `= x₁  =  ${E.x1}` },
      { lbl: '∂z/∂w₂',  eq: `= x₂  =  ${E.x2}` },
      { lbl: '∂L/∂w₁',  eq: `= δ × x₁  =  ${E.delta} × ${E.x1}  =  ${E.gw1}` },
      { lbl: '∂L/∂w₂',  eq: `= δ × x₂  =  ${E.delta} × ${E.x2}  =  ${E.gw2}` },
    ],
    note: 'Bigger input → bigger gradient → that connection gets a larger update. Makes intuitive sense!',
    hl: 'bwd-weight',
  },
  {
    title: 'Step 6 — Through Hidden Layers',
    subtitle: 'The error signal δ propagates backward layer by layer',
    body: [
      'In a multi-layer network, after computing δ at the output we propagate it backward. The transposed weight matrix routes the error signal to each hidden neuron, and σ\'(z) gates it.',
    ],
    math: [
      { lbl: 'Layer rule', eq: "δ^(l)  =  (W^(l+1))ᵀ · δ^(l+1)  ⊙  σ'(z^(l))" },
      { lbl: 'W^T routes', eq: 'gradient back proportional to each weight value' },
      { lbl: '⊙  gates',  eq: "by local sensitivity σ'(z^(l)) ≤ 0.25" },
      { lbl: 'Vanishing', eq: "σ'≤0.25 per layer → deep networks learn slowly at early layers" },
    ],
    note: 'ReLU fixes this: its gradient is exactly 1 for z>0, so signal flows freely through layers.',
    hl: 'bwd-hidden',
  },
  {
    title: 'Step 7 — Weight Update: Gradient Descent',
    subtitle: 'Subtract α × gradient from each weight',
    body: [
      'After computing all gradients we update every weight: subtract α (the learning rate) times the gradient. Negative gradient → increasing w reduces loss → gradient descent increases w.',
    ],
    math: [
      { lbl: 'Rule',  eq: 'w  ←  w − α · ∂L/∂w' },
      { lbl: 'w₁',   eq: `${E.w1} − ${E.lr}×(${E.gw1})  =  ${E.w1n}` },
      { lbl: 'w₂',   eq: `${E.w2} − ${E.lr}×(${E.gw2})  =  ${E.w2n}` },
      { lbl: 'b',    eq: `${E.b}  − ${E.lr}×(${E.gb})  =  ${E.bn}` },
    ],
    note: 'α=0.01 is tiny on purpose. Large α can overshoot the minimum and cause the loss to diverge.',
    hl: 'update',
  },
  {
    title: 'Full Picture — One Complete Epoch',
    subtitle: 'Forward → Loss → Backward → Update → Repeat',
    body: [
      'One epoch processes every training sample: forward pass computes ŷ and L; backward pass computes all ∂L/∂w; the optimizer updates all weights. Repeat for hundreds of epochs.',
      'Adam, RMSprop etc. use smarter update rules, but the gradients all come from the same backprop.',
    ],
    math: [
      { lbl: 'Forward',  eq: 'x → z^(1) → a^(1) → z^(2) → ŷ → L' },
      { lbl: 'Backward', eq: "δ_out → δ^(l) → ∂L/∂W^(l)   ∀l" },
      { lbl: 'Update',   eq: 'W ← W − α · ∇_W L' },
      { lbl: 'Repeat',   eq: 'for every sample, every epoch' },
    ],
    note: 'Training = gradient descent on the loss surface. Each epoch, the network gets a little better.',
    hl: 'full',
  },
];

// ─── Color palette ────────────────────────────────────────────────────────────
const C = {
  fwd:  '#22c55e',   // green — forward pass
  bwd:  '#ef4444',   // red   — backward pass / gradients
  upd:  '#a855f7',   // purple — weight updates
  loss: '#f59e0b',   // amber  — loss
  inp:  '#60a5fa',   // blue   — input nodes
  hid:  '#22c55e',   // green  — hidden nodes
  out:  '#ef4444',   // red    — output node
  dim:  '#1e293b',   // dark slate — inactive
  dimStroke: '#334155',
};

// ─── Node + edge configurations ──────────────────────────────────────────────
// Network layout (SVG 520 × 340)
const NX = {
  i1:   { x: 70,  y: 100 },
  i2:   { x: 70,  y: 240 },
  h1:   { x: 230, y: 100 },
  h2:   { x: 230, y: 240 },
  out:  { x: 390, y: 170 },
};
const R = 28; // node radius

// Highlight config: for each `hl` key, what's active
const HIGHLIGHT = {
  intro:      { nodes: [], fwdEdges: [], bwdEdges: [], updEdges: [], lossActive: false },
  'fwd-z':    { nodes: ['i1','i2','h1','h2'], fwdEdges: ['i1-h1','i1-h2','i2-h1','i2-h2'], bwdEdges: [], updEdges: [], lossActive: false },
  'fwd-a':    { nodes: ['h1','h2','out'], fwdEdges: ['i1-h1','i1-h2','i2-h1','i2-h2','h1-o','h2-o'], bwdEdges: [], updEdges: [], lossActive: false },
  loss:       { nodes: ['out'], fwdEdges: ['h1-o','h2-o'], bwdEdges: [], updEdges: [], lossActive: true },
  chain:      { nodes: ['i1','h1','out'], fwdEdges: ['i1-h1','h1-o'], bwdEdges: ['h1-o','i1-h1'], updEdges: [], lossActive: true },
  'bwd-output': { nodes: ['out'], fwdEdges: [], bwdEdges: ['h1-o','h2-o'], updEdges: [], lossActive: true },
  'bwd-weight': { nodes: ['i1','i2','h1'], fwdEdges: [], bwdEdges: ['i1-h1','i2-h1','h1-o'], updEdges: [], lossActive: true },
  'bwd-hidden': { nodes: ['i1','i2','h1','h2','out'], fwdEdges: [], bwdEdges: ['i1-h1','i1-h2','i2-h1','i2-h2','h1-o','h2-o'], updEdges: [], lossActive: true },
  update:     { nodes: ['i1','i2','h1','h2','out'], fwdEdges: [], bwdEdges: [], updEdges: ['i1-h1','i1-h2','i2-h1','i2-h2','h1-o','h2-o'], lossActive: false },
  full:       { nodes: ['i1','i2','h1','h2','out'], fwdEdges: ['i1-h1','i1-h2','i2-h1','i2-h2','h1-o','h2-o'], bwdEdges: ['i1-h1','i1-h2','i2-h1','i2-h2','h1-o','h2-o'], updEdges: [], lossActive: true },
};

const EDGES = [
  { id: 'i1-h1', from: 'i1', to: 'h1', wLabel: 'w₁=0.8' },
  { id: 'i1-h2', from: 'i1', to: 'h2', wLabel: 'w₃'     },
  { id: 'i2-h1', from: 'i2', to: 'h1', wLabel: 'w₂=−0.4'},
  { id: 'i2-h2', from: 'i2', to: 'h2', wLabel: 'w₄'     },
  { id: 'h1-o',  from: 'h1', to: 'out', wLabel: 'u₁'    },
  { id: 'h2-o',  from: 'h2', to: 'out', wLabel: 'u₂'    },
];

// Value overlays per step
const VALUE_OVERLAYS = {
  'fwd-z':    [{ x: 230, y: 86,  text: 'z=0.38', color: C.fwd }, { x: 70, y: 86, text: 'x₁=0.5', color: C.inp }, { x: 70, y: 226, text: 'x₂=0.3', color: C.inp }],
  'fwd-a':    [{ x: 230, y: 86,  text: 'a≈0.594', color: C.hid }, { x: 390, y: 156, text: 'ŷ=?', color: C.out }],
  loss:       [{ x: 390, y: 156, text: 'ŷ≈0.594', color: C.out }],
  chain:      [{ x: 390, y: 156, text: 'δ=ŷ−y', color: C.bwd }],
  'bwd-output': [{ x: 390, y: 156, text: 'δ=−0.406', color: C.bwd }],
  'bwd-weight': [{ x: 150, y: 88, text: '∂L/∂w₁=−0.203', color: C.bwd }],
  'bwd-hidden': [{ x: 230, y: 86, text: 'δ^(l)', color: C.bwd }],
  update:     [{ x: 150, y: 88, text: 'w₁: 0.800→0.802', color: C.upd }],
  full:       [{ x: 230, y: 86, text: 'fwd+bwd', color: '#a78bfa' }],
};

// ─── SVG Network Diagram ─────────────────────────────────────────────────────
function NetworkSVG({ hl }) {
  const cfg = HIGHLIGHT[hl] || HIGHLIGHT.intro;

  function edgeColor(edgeId) {
    if (cfg.updEdges.includes(edgeId)) return C.upd;
    if (cfg.bwdEdges.includes(edgeId)) return C.bwd;
    if (cfg.fwdEdges.includes(edgeId)) return C.fwd;
    return C.dimStroke;
  }
  function edgeOpacity(edgeId) {
    const active = cfg.fwdEdges.includes(edgeId) || cfg.bwdEdges.includes(edgeId) || cfg.updEdges.includes(edgeId);
    return active ? 1 : 0.18;
  }
  function edgeWidth(edgeId) {
    const active = cfg.fwdEdges.includes(edgeId) || cfg.bwdEdges.includes(edgeId) || cfg.updEdges.includes(edgeId);
    return active ? 2.5 : 1.5;
  }
  function nodeColor(nodeId) {
    if (!cfg.nodes.includes(nodeId)) return C.dim;
    if (nodeId.startsWith('i')) return '#1e3a5f';
    if (nodeId.startsWith('h')) return '#1a3320';
    return '#3b1f1f';
  }
  function nodeStroke(nodeId) {
    if (!cfg.nodes.includes(nodeId)) return C.dimStroke;
    if (nodeId.startsWith('i')) return C.inp;
    if (nodeId.startsWith('h')) return C.hid;
    return C.out;
  }
  function nodeOpacity(nodeId) {
    return cfg.nodes.includes(nodeId) ? 1 : 0.3;
  }

  const overlays = VALUE_OVERLAYS[hl] || [];
  const lossColor = cfg.lossActive ? C.loss : C.dimStroke;
  const lossOpacity = cfg.lossActive ? 1 : 0.2;

  // Arrow marker helper
  function arrowHead(id, color) {
    return (
      <marker key={id} id={id} markerWidth="8" markerHeight="6" refX="6" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill={color} />
      </marker>
    );
  }

  // Draw edge between two nodes
  function drawEdge(edge) {
    const from = NX[edge.from], to = NX[edge.to];
    const dx = to.x - from.x, dy = to.y - from.y;
    const len = Math.sqrt(dx*dx + dy*dy);
    const ux = dx/len, uy = dy/len;
    const x1 = from.x + ux*R, y1 = from.y + uy*R;
    const x2 = to.x   - ux*R, y2 = to.y   - uy*R;
    const col = edgeColor(edge.id);
    const markerId = `arrow-${edge.id}-${hl}`;
    const isBwd = cfg.bwdEdges.includes(edge.id) && !cfg.fwdEdges.includes(edge.id);
    const isFull = cfg.fwdEdges.includes(edge.id) && cfg.bwdEdges.includes(edge.id);

    return (
      <g key={edge.id} opacity={edgeOpacity(edge.id)}>
        <defs>{arrowHead(markerId, col)}</defs>
        {/* Edge line */}
        <line
          x1={isBwd ? x2 : x1} y1={isBwd ? y2 : y1}
          x2={isBwd ? x1 : x2} y2={isBwd ? y1 : y2}
          stroke={col}
          strokeWidth={edgeWidth(edge.id)}
          markerEnd={`url(#${markerId})`}
          strokeDasharray={isFull ? '5 3' : 'none'}
        />
        {/* Weight label on active main path edges only */}
        {(edge.id === 'i1-h1' || edge.id === 'h1-o') && cfg.fwdEdges.includes(edge.id) && (
          <text
            x={(from.x + to.x)/2}
            y={(from.y + to.y)/2 - 9}
            fill={col}
            fontSize="11"
            textAnchor="middle"
            fontFamily="monospace"
          >
            {edge.wLabel}
          </text>
        )}
      </g>
    );
  }

  return (
    <svg viewBox="0 0 520 340" className="w-full h-full" style={{ maxHeight: '100%' }}>
      {/* Background grid dots */}
      {Array.from({ length: 6 }, (_, i) =>
        Array.from({ length: 5 }, (_, j) => (
          <circle key={`${i}-${j}`} cx={40 + i*88} cy={30 + j*68} r="1.5" fill="#1e293b" />
        ))
      )}

      {/* Layer labels */}
      <text x="70"  y="30" fill="#334155" fontSize="11" textAnchor="middle" fontFamily="monospace">Input</text>
      <text x="230" y="30" fill="#334155" fontSize="11" textAnchor="middle" fontFamily="monospace">Hidden</text>
      <text x="390" y="30" fill="#334155" fontSize="11" textAnchor="middle" fontFamily="monospace">Output</text>

      {/* Edges */}
      {EDGES.map(drawEdge)}

      {/* Loss connection */}
      <g opacity={lossOpacity}>
        <defs>
          <marker id={`arrow-loss-${hl}`} markerWidth="8" markerHeight="6" refX="6" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill={lossColor} />
          </marker>
        </defs>
        <line
          x1={NX.out.x + R} y1={NX.out.y}
          x2={456}           y2={170}
          stroke={lossColor}
          strokeWidth={cfg.lossActive ? 2.5 : 1.5}
          markerEnd={`url(#arrow-loss-${hl})`}
        />
      </g>

      {/* Loss box */}
      <g opacity={lossOpacity}>
        <rect x="458" y="150" width="52" height="40" rx="8"
          fill={cfg.lossActive ? '#3b2f0f' : C.dim}
          stroke={lossColor}
          strokeWidth={cfg.lossActive ? 2 : 1}
        />
        <text x="484" y="165" fill={lossColor} fontSize="11" textAnchor="middle" fontFamily="monospace" fontWeight="bold">Loss</text>
        <text x="484" y="181" fill={lossColor} fontSize="10" textAnchor="middle" fontFamily="monospace">L</text>
      </g>

      {/* Nodes */}
      {Object.entries(NX).map(([id, pos]) => (
        <g key={id} opacity={nodeOpacity(id)}>
          <circle cx={pos.x} cy={pos.y} r={R}
            fill={nodeColor(id)}
            stroke={nodeStroke(id)}
            strokeWidth={cfg.nodes.includes(id) ? 2.5 : 1.5}
          />
          {/* Glow ring */}
          {cfg.nodes.includes(id) && (
            <circle cx={pos.x} cy={pos.y} r={R + 6}
              fill="none"
              stroke={nodeStroke(id)}
              strokeWidth="1"
              opacity="0.25"
            />
          )}
          {/* Node label */}
          <text x={pos.x} y={pos.y + 1} fill={cfg.nodes.includes(id) ? nodeStroke(id) : C.dimStroke}
            fontSize="14" textAnchor="middle" dominantBaseline="middle"
            fontFamily="monospace" fontWeight="bold">
            {id === 'i1' ? 'x₁' : id === 'i2' ? 'x₂' : id === 'h1' ? 'h₁' : id === 'h2' ? 'h₂' : 'ŷ'}
          </text>
        </g>
      ))}

      {/* Value overlays */}
      {overlays.map((ov, i) => (
        <g key={i}>
          <rect x={ov.x - 4} y={ov.y - 12} width={(ov.text.length * 6.5) + 8} height="17"
            rx="4" fill="#0f172a" opacity="0.85" />
          <text x={ov.x} y={ov.y} fill={ov.color}
            fontSize="11" fontFamily="monospace" fontWeight="bold">
            {ov.text}
          </text>
        </g>
      ))}

      {/* Chain rule inset for 'chain' step */}
      {hl === 'chain' && (
        <g transform="translate(30, 290)">
          <rect x="0" y="0" width="460" height="40" rx="8" fill="#1e293b" stroke="#334155" strokeWidth="1" />
          {[
            { x: 16,  text: '∂L/∂w',  color: C.bwd  },
            { x: 78,  text: '=',       color: '#94a3b8' },
            { x: 92,  text: '∂L/∂a',  color: '#a78bfa' },
            { x: 153, text: '·',       color: '#94a3b8' },
            { x: 162, text: '∂a/∂z',  color: C.fwd   },
            { x: 220, text: '·',       color: '#94a3b8' },
            { x: 229, text: '∂z/∂w',  color: C.inp   },
            { x: 293, text: '=',       color: '#94a3b8' },
            { x: 305, text: 'δ',       color: C.bwd   },
            { x: 320, text: '·',       color: '#94a3b8' },
            { x: 329, text: "σ'(z)",   color: C.fwd   },
            { x: 376, text: '·',       color: '#94a3b8' },
            { x: 385, text: 'x',       color: C.inp   },
          ].map(({ x, text, color }) => (
            <text key={x} x={x} y="25" fill={color} fontSize="12" fontFamily="monospace" fontWeight="bold">{text}</text>
          ))}
        </g>
      )}

      {/* Full epoch labels */}
      {hl === 'full' && (
        <>
          <text x="150" y="310" fill={C.fwd}  fontSize="11" textAnchor="middle" fontFamily="monospace">→ Forward</text>
          <text x="310" y="310" fill={C.bwd}  fontSize="11" textAnchor="middle" fontFamily="monospace">← Backward</text>
          <text x="450" y="310" fill={C.loss} fontSize="11" textAnchor="middle" fontFamily="monospace">Loss L</text>
        </>
      )}

      {/* Step direction indicator */}
      {(hl === 'fwd-z' || hl === 'fwd-a' || hl === 'loss') && (
        <text x="260" y="310" fill={C.fwd} fontSize="11" textAnchor="middle" fontFamily="monospace" opacity="0.7">
          ── Forward pass ──▶
        </text>
      )}
      {(hl === 'bwd-output' || hl === 'bwd-weight' || hl === 'bwd-hidden') && (
        <text x="260" y="310" fill={C.bwd} fontSize="11" textAnchor="middle" fontFamily="monospace" opacity="0.7">
          ◀── Backward pass ──
        </text>
      )}
      {hl === 'update' && (
        <text x="260" y="310" fill={C.upd} fontSize="11" textAnchor="middle" fontFamily="monospace" opacity="0.7">
          ↕ Weight Update
        </text>
      )}
    </svg>
  );
}

// ─── Math block ───────────────────────────────────────────────────────────────
function MathBlock({ lines }) {
  return (
    <div className="rounded-xl border border-white/5 overflow-hidden">
      <div className="bg-slate-900/80 px-3 py-1.5 border-b border-white/5">
        <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Mathematics</span>
      </div>
      <div className="bg-slate-950/60 p-4 space-y-1.5">
        {lines.map((line, i) => {
          const isBlank = !line.lbl.trim() && !line.eq.trim();
          if (isBlank) return <div key={i} className="h-1" />;
          const isArrow = line.eq.startsWith('↑') || line.eq.startsWith('←') || line.eq.startsWith('→');
          return (
            <div key={i} className="flex gap-3 items-baseline">
              <span className="text-[11px] text-slate-500 font-mono w-24 shrink-0 text-right leading-relaxed">
                {line.lbl.trim() ? line.lbl : ''}
              </span>
              <span
                className="font-mono text-sm leading-relaxed"
                style={{ color: isArrow ? '#475569' : i === 0 ? '#a78bfa' : '#e2e8f0' }}
              >
                {line.eq}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function BackpropExplainer({ onClose }) {
  const [step, setStep] = useState(0);
  const total = STEPS.length;
  const current = STEPS[step];

  // Keyboard navigation — runs only while this modal is mounted
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown')  { e.stopPropagation(); setStep(s => Math.min(total - 1, s + 1)); }
      if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')    { e.stopPropagation(); setStep(s => Math.max(0, s - 1)); }
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler, { capture: true });
    return () => window.removeEventListener('keydown', handler, { capture: true });
  }, [onClose, total]);

  // Step colour ring
  const stepColors = ['#60a5fa','#22c55e','#22c55e','#f59e0b','#a78bfa','#ef4444','#ef4444','#ef4444','#a855f7','#a78bfa'];
  const ringColor = stepColors[step] || '#60a5fa';

  return (
    <div
      className="fixed inset-0 z-50 bg-slate-950/96 backdrop-blur-sm flex flex-col"
      onClick={e => e.target === e.currentTarget && onClose()}
    >
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-white/5 shrink-0">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl" style={{ background: ringColor + '22', border: `1px solid ${ringColor}44` }}>
            <Brain size={18} style={{ color: ringColor }} />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">Backpropagation — Interactive Walkthrough</h2>
            <p className="text-[11px] text-slate-500">Step {step + 1} of {total}  ·  Use ← → arrow keys to navigate</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-all"
          aria-label="Close explainer"
        >
          <X size={16} />
        </button>
      </header>

      {/* ── Body ───────────────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">

        {/* LEFT — Network Diagram */}
        <div className="flex-1 flex flex-col min-w-0 border-r border-white/5">
          <div className="flex-1 flex items-center justify-center p-6 min-h-0">
            <div className="w-full h-full max-w-[540px]">
              <NetworkSVG hl={current.hl} />
            </div>
          </div>

          {/* Step pill row */}
          <div className="shrink-0 px-6 pb-4">
            <div
              className="rounded-xl p-3 border text-xs font-mono leading-relaxed"
              style={{ background: ringColor + '0d', borderColor: ringColor + '33', color: ringColor }}
            >
              <span className="text-slate-500 font-sans mr-2">Direction:</span>
              {['intro'].includes(current.hl) && 'None — overview'}
              {['fwd-z','fwd-a','loss'].includes(current.hl) && '→  Forward pass  (input to output)'}
              {['chain'].includes(current.hl) && '↔  Both directions (chain rule)'}
              {['bwd-output','bwd-weight','bwd-hidden'].includes(current.hl) && '←  Backward pass  (loss to weights)'}
              {['update'].includes(current.hl) && '↕  Weight update  (gradient descent)'}
              {['full'].includes(current.hl) && '↔  Complete epoch  (forward + backward + update)'}
            </div>
          </div>
        </div>

        {/* RIGHT — Explanation */}
        <div className="w-[420px] shrink-0 flex flex-col overflow-y-auto p-6 gap-5">

          {/* Title */}
          <div>
            <div
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest mb-2"
              style={{ background: ringColor + '18', color: ringColor }}
            >
              Step {step + 1} of {total}
            </div>
            <h3 className="text-xl font-black text-white leading-tight">{current.title}</h3>
            <p className="text-sm text-slate-400 mt-0.5">{current.subtitle}</p>
          </div>

          {/* Concept text */}
          <div className="space-y-2">
            {current.body.map((para, i) => (
              <p key={i} className="text-sm text-slate-300 leading-relaxed">{para}</p>
            ))}
          </div>

          {/* Math */}
          <MathBlock lines={current.math} />

          {/* Note */}
          <div className="rounded-xl p-3 bg-amber-500/5 border border-amber-500/20 flex gap-2.5 items-start">
            <span className="text-amber-400 text-base shrink-0 mt-0.5">💡</span>
            <p className="text-xs text-amber-200/80 leading-relaxed italic">{current.note}</p>
          </div>

          {/* Numerical example badge */}
          {step > 0 && step < total - 1 && (
            <div className="rounded-xl p-3 bg-slate-900 border border-white/5">
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">Running Example</p>
              <div className="flex flex-wrap gap-2 text-[11px] font-mono">
                {[
                  { k: 'x₁', v: E.x1, c: C.inp },
                  { k: 'x₂', v: E.x2, c: C.inp },
                  { k: 'w₁', v: E.w1, c: '#22c55e' },
                  { k: 'w₂', v: E.w2, c: '#22c55e' },
                  { k: 'b',  v: E.b,  c: '#22c55e' },
                  { k: 'z',  v: E.z,  c: '#a78bfa' },
                  { k: 'ŷ',  v: E.a,  c: C.out },
                  { k: 'L',  v: E.loss, c: C.loss },
                  { k: 'δ',  v: E.delta, c: C.bwd },
                ].map(({ k, v, c }) => (
                  <span key={k} className="px-2 py-0.5 rounded-md" style={{ background: c + '18', color: c }}>
                    {k}={v}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Footer navigation ──────────────────────────────────────── */}
      <footer className="shrink-0 flex items-center gap-4 px-6 py-3 border-t border-white/5">
        <button
          onClick={() => setStep(s => Math.max(0, s - 1))}
          disabled={step === 0}
          className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-semibold bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-all text-slate-300"
        >
          <ChevronLeft size={16} />Prev
        </button>

        {/* Dot indicators */}
        <div className="flex-1 flex items-center justify-center gap-1.5 flex-wrap">
          {STEPS.map((s, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              aria-label={`Go to step ${i + 1}: ${s.title}`}
              className="rounded-full transition-all duration-200"
              style={{
                width: i === step ? 20 : 8,
                height: 8,
                background: i === step ? ringColor : i < step ? ringColor + '60' : '#334155',
              }}
            />
          ))}
        </div>

        <button
          onClick={() => setStep(s => Math.min(total - 1, s + 1))}
          disabled={step === total - 1}
          className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-semibold text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          style={{ background: step === total - 1 ? '#334155' : ringColor }}
        >
          Next<ChevronRight size={16} />
        </button>
      </footer>
    </div>
  );
}
