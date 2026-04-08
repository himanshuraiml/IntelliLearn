"use client";
import React from 'react';
import { Terminal } from 'lucide-react';

const CODE_MAP = {
  linear: (p) => `// TensorFlow.js — Linear Regression
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({
  optimizer: tf.train.sgd(${p.learningRate}),
  loss: 'meanSquaredError'
});
await model.fit(xs, ys, { epochs: ${p.epochs} });
// y = weight * x + bias`,

  poly: (p) => `// Polynomial Regression (degree ${p.degree})
// Feature expansion: x → [x, x², ..., x^${p.degree}]
const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [${p.degree}] }));
model.add(tf.layers.dense({ units: 1 }));
model.compile({
  optimizer: tf.train.adam(${p.learningRate}),
  loss: 'meanSquaredError'
});
await model.fit(xs, ys, { epochs: ${p.epochs} });`,

  ridge: (p) => `// Ridge Regression — L2 Regularization
// Loss = MSE + λ * Σ(w²)
let w = tf.variable(tf.randomNormal([1, 1]));
let b = tf.variable(tf.zeros([1]));
const optimizer = tf.train.sgd(${p.learningRate});
for (let epoch = 0; epoch < ${p.epochs}; epoch++) {
  optimizer.minimize(() => {
    const preds = xs.matMul(w).add(b);
    const mse = tf.losses.meanSquaredError(ys, preds);
    const l2 = tf.mul(0.1, tf.sum(tf.square(w)));
    return mse.add(l2);
  });
}`,

  logistic: (p) => `// TensorFlow.js — Logistic Regression
// Input: [x, y], Output: class probability
const model = tf.sequential();
model.add(tf.layers.dense({
  units: 1, activation: 'sigmoid', inputShape: [2]
}));
model.compile({
  optimizer: tf.train.adam(${p.learningRate}),
  loss: 'binaryCrossentropy'
});
await model.fit(xs, ys, { epochs: ${p.epochs} });
// Decision boundary: w1*x + w2*y + b = 0`,

  knn: (p) => `// K-Nearest Neighbors (k=${p.k})
// Pure JavaScript — no training needed!
function predict(trainData, point, k = ${p.k}) {
  const distances = trainData
    .map(p => ({ ...p, dist: euclidean(p, point) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, k);
  // Majority vote
  const counts = {};
  distances.forEach(p => counts[p.label] = (counts[p.label]||0)+1);
  return Object.keys(counts).reduce((a,b) => counts[a]>counts[b]?a:b);
}`,

  naiveBayes: () => `// Gaussian Naive Bayes
// P(y|x) ∝ P(y) * P(x1|y) * P(x2|y)
function gaussianPDF(x, mean, variance) {
  return (1 / Math.sqrt(2 * Math.PI * variance))
    * Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
}
function predict(stats, x, y) {
  let bestClass, maxProb = -Infinity;
  for (const [cls, s] of Object.entries(stats)) {
    const logProb = Math.log(s.prior)
      + Math.log(gaussianPDF(x, s.mean.x, s.variance.x))
      + Math.log(gaussianPDF(y, s.mean.y, s.variance.y));
    if (logProb > maxProb) { maxProb = logProb; bestClass = cls; }
  }
  return bestClass;
}`,

  svm: (p) => `// Support Vector Machine (SGD, C=${p.C})
// Hinge loss + L2 regularization
let w = [0, 0], b = 0;
for (let epoch = 0; epoch < ${p.epochs}; epoch++) {
  data.forEach(({ x, y, label }) => {
    const yi = label === 0 ? -1 : 1;
    const margin = yi * (w[0]*x + w[1]*y + b);
    if (margin < 1) {
      w[0] -= lr * (2*lambda*w[0] - C*yi*x);
      w[1] -= lr * (2*lambda*w[1] - C*yi*y);
      b    += lr * C * yi;
    } else {
      w[0] -= lr * 2*lambda*w[0];
      w[1] -= lr * 2*lambda*w[1];
    }
  });
}
// Boundary: w[0]*x + w[1]*y + b = 0`,

  decisionTree: (p) => `// Decision Tree (CART, maxDepth=${p.maxDepth})
function buildTree(data, depth=0) {
  if (depth >= ${p.maxDepth} || isPure(data)) return toLeaf(data);
  const { feature, threshold } = bestSplit(data);  // min Gini
  return {
    feature, threshold,
    left:  buildTree(data.filter(d=>d[feature]<=threshold), depth+1),
    right: buildTree(data.filter(d=>d[feature]>threshold),  depth+1),
  };
}
// Gini Impurity = 1 - Σ(pᵢ²)
function gini(group) {
  const classes = [...new Set(group.map(d=>d.label))];
  return 1 - classes.reduce((s,c) => {
    const p = group.filter(d=>d.label===c).length / group.length;
    return s + p*p;
  }, 0);
}`,

  randomForest: (p) => `// Random Forest (${p.nTrees} trees, depth=${p.maxDepth})
// Ensemble of Decision Trees with bootstrap sampling
const trees = [];
for (let t = 0; t < ${p.nTrees}; t++) {
  // Bootstrap sample (random with replacement)
  const sample = Array.from({ length: data.length },
    () => data[Math.floor(Math.random() * data.length)]);
  trees.push(buildTree(sample, 0, ${p.maxDepth}));
}
function predict(x, y) {
  const votes = {};
  trees.forEach(tree => {
    const label = predictTree(tree, {x, y});
    votes[label] = (votes[label] || 0) + 1;
  });
  return Object.keys(votes).reduce((a,b) => votes[a]>votes[b]?a:b);
}`,

  kmeans: (p) => `// K-Means Clustering (k=${p.k})
let centroids = randomInit(data, ${p.k});
while (changed) {
  // 1. Assignment step
  data.forEach(pt => {
    pt.cluster = nearest(pt, centroids);
  });
  // 2. Update step — move centroids to mean
  centroids = centroids.map((_, k) => {
    const pts = data.filter(p => p.cluster === k);
    return { x: mean(pts, 'x'), y: mean(pts, 'y') };
  });
}`,

  nn: (p) => `// Neural Network MLP (TensorFlow.js)
const model = tf.sequential();
model.add(tf.layers.dense({ units: 4, activation: 'relu', inputShape: [2] }));
model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
model.compile({
  optimizer: tf.train.adam(${p.learningRate}),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy']
});
await model.fit(xs, ys, { epochs: ${p.epochs} });`,

  dnn: (p) => `// Deep Neural Network (TensorFlow.js)
const model = tf.sequential();
model.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [2] }));
model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
model.compile({
  optimizer: tf.train.adam(${p.learningRate}),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy']
});
await model.fit(xs, ys, { epochs: ${p.epochs} });`,
};

const CodeSnippet = ({ activeTab, params }) => {
  const getCode = () => {
    const fn = CODE_MAP[activeTab];
    return fn ? fn(params) : `// ${activeTab} implementation`;
  };

  return (
    <div className="bg-slate-950 rounded-xl border border-white/5 overflow-hidden font-mono text-sm">
      <div className="bg-white/5 px-4 py-2 border-b border-white/5 flex items-center gap-2">
        <Terminal size={14} className="text-brand-500" />
        <span className="text-xs text-slate-400">Implementation — {activeTab}</span>
      </div>
      <pre className="p-4 text-slate-300 overflow-x-auto text-xs leading-relaxed">
        <code>{getCode()}</code>
      </pre>
    </div>
  );
};

export default CodeSnippet;
