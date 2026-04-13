"use client";
import React from 'react';

/**
 * Renders the mathematical formula for the current algorithm,
 * with learned parameter values substituted in (when modelResult is available).
 */

function fmt(v, decimals = 3) {
  if (v == null) return '?';
  const n = parseFloat(v);
  if (isNaN(n)) return '?';
  return n >= 0 ? `+${n.toFixed(decimals)}` : n.toFixed(decimals);
}
function fmtPlain(v, decimals = 3) {
  if (v == null) return '?';
  const n = parseFloat(v);
  if (isNaN(n)) return '?';
  return n.toFixed(decimals);
}

function getFormulaData(algo, params, modelResult) {
  const m = modelResult;

  switch (algo) {
    case 'linear': return {
      label: 'Learned Equation',
      main:  m?.weight != null
        ? `ŷ = ${fmtPlain(m.weight, 3)}·x ${fmt(m.bias, 3)}`
        : `ŷ = w·x + b`,
      sub:   `Minimises: Σ(y − ŷ)²   |   LR = ${params.learningRate}`,
      color: '#38bdf8',
    };

    case 'poly': return {
      label: 'Polynomial Model',
      main:  `ŷ = Σ wₖ·xᵏ   (degree ${params.degree})`,
      sub:   `Features: [x, x², ..., x^${params.degree}]   |   LR = ${params.learningRate}`,
      color: '#a78bfa',
    };

    case 'ridge': return {
      label: 'Ridge (L2) Equation',
      main:  m?.weight != null
        ? `ŷ = ${fmtPlain(m.weight, 3)}·x ${fmt(m.bias, 3)}`
        : `ŷ = w·x + b`,
      sub:   `Loss = MSE + λ·Σ(w²)   |   λ = 0.1`,
      color: '#34d399',
    };

    case 'logistic': return {
      label: 'Decision Boundary',
      main:  m?.weights
        ? `P(y=1) = σ(${fmtPlain(m.weights[0], 3)}·x ${fmt(m.weights[1], 3)}·y ${fmt(m.bias, 3)})`
        : `P(y=1) = σ(w₁·x + w₂·y + b)`,
      sub:   `σ(z) = 1/(1+e⁻ᶻ)   |   Boundary: w₁x + w₂y + b = 0`,
      color: '#f472b6',
    };

    case 'svm': return {
      label: 'SVM Hyperplane',
      main:  m?.weights
        ? `f(x,y) = ${fmtPlain(m.weights[0], 3)}·x ${fmt(m.weights[1], 3)}·y ${fmt(m.bias, 3)}`
        : `f(x,y) = w₁·x + w₂·y + b`,
      sub:   `Sign(f) → class   |   Margin = 2/‖w‖   |   C = ${params.C}`,
      color: '#f59e0b',
    };

    case 'knn': return {
      label: 'Distance Metric',
      main:  `d(p,q) = √((pₓ−qₓ)² + (pᵧ−qᵧ)²)`,
      sub:   `k = ${params.k}  →  majority vote among ${params.k} nearest neighbours`,
      color: '#60a5fa',
    };

    case 'naiveBayes': return {
      label: "Bayes' Theorem",
      main:  `P(C|x,y) ∝ P(C)·P(x|C)·P(y|C)`,
      sub:   `Gaussian: P(x|C) = (1/√2πσ²)·exp(−(x−μ)²/2σ²)`,
      color: '#10b981',
    };

    case 'decisionTree': return {
      label: 'Gini Impurity Split',
      main:  `Gini = 1 − Σ pᵢ²   (maxDepth = ${params.maxDepth})`,
      sub:   m?.accuracy != null
        ? `Train accuracy: ${(m.accuracy * 100).toFixed(1)}%`
        : `Split on min Gini across all features & thresholds`,
      color: '#fb923c',
    };

    case 'randomForest': return {
      label: 'Ensemble Vote',
      main:  `ŷ = majority{ tree₁(x), …, tree_${params.nTrees}(x) }`,
      sub:   `${params.nTrees} bootstrap trees, maxDepth = ${params.maxDepth}`,
      color: '#f97316',
    };

    case 'kmeans': return {
      label: 'Centroid Update',
      main:  `μₖ = (1/|Cₖ|) · Σ xᵢ   (k = ${params.k})`,
      sub:   `Assign: argmin‖xᵢ − μₖ‖   →   Update centroids`,
      color: '#e879f9',
    };

    case 'nn':
    case 'dnn': {
      const layers = algo === 'dnn' ? [2, 8, 8, 4, 1] : [2, 4, 4, 1];
      return {
        label: 'MLP Architecture',
        main:  layers.join(' → '),
        sub:   `Activation: ReLU (hidden), Sigmoid (output)   |   LR = ${params.learningRate}`,
        color: '#818cf8',
      };
    }

    case 'pca': return {
      label: 'Principal Components',
      main:  m?.explainedVar1 != null
        ? `PC1 explains ${(m.explainedVar1 * 100).toFixed(1)}%   |   PC2 explains ${(m.explainedVar2 * 100).toFixed(1)}%`
        : `X_pca = X · V   (V = eigenvectors of Cov(X))`,
      sub:   `Cov = (XᵀX)/(n−1)   |   EVD → sorted eigenvectors`,
      color: '#34d399',
    };

    default: return null;
  }
}

const MathFormula = ({ algo, params, modelResult }) => {
  const formula = getFormulaData(algo, params, modelResult);
  if (!formula) return null;

  return (
    <div className="bg-slate-900/80 rounded-xl border border-white/5 p-4">
      <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-3">
        {formula.label}
      </h3>
      <div
        className="font-mono text-sm font-semibold px-3 py-2.5 rounded-lg mb-2 leading-relaxed"
        style={{ background: formula.color + '12', color: formula.color, border: `1px solid ${formula.color}25` }}
      >
        {formula.main}
      </div>
      <p className="text-[11px] text-slate-500 leading-relaxed font-mono">{formula.sub}</p>
    </div>
  );
};

export default MathFormula;
