"use client";
import React, { useState } from 'react';
import { Terminal } from 'lucide-react';

// ── JavaScript / TF.js snippets ──────────────────────────────────────────────
const JS_MAP = {
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
  loss: 'binaryCrossentropy',
  metrics: ['accuracy']
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

// ── Python / scikit-learn / Keras snippets ───────────────────────────────────
const PY_MAP = {
  linear: (p) => `# scikit-learn — Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

model = LinearRegression()
model.fit(X, y)          # X shape: (n, 1)

# Equivalent TF/Keras gradient descent:
# from tensorflow import keras
# model = keras.Sequential([keras.layers.Dense(1, input_shape=(1,))])
# model.compile(optimizer=keras.optimizers.SGD(${p.learningRate}),
#               loss='mse')
# model.fit(X, y, epochs=${p.epochs})

print(f"weight = {model.coef_[0]:.4f}")
print(f"bias   = {model.intercept_[0]:.4f}")
# y = weight * x + bias`,

  poly: (p) => `# scikit-learn — Polynomial Regression (degree ${p.degree})
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    PolynomialFeatures(degree=${p.degree}),
    LinearRegression()
)
model.fit(X, y)          # X shape: (n, 1)
y_pred = model.predict(X_test)

# Keras equivalent:
# from tensorflow import keras
# model = keras.Sequential([
#     keras.layers.Dense(16, activation='relu', input_shape=(${p.degree},)),
#     keras.layers.Dense(1)
# ])
# model.compile(optimizer='adam', loss='mse')
# model.fit(X_poly, y, epochs=${p.epochs})`,

  ridge: (p) => `# scikit-learn — Ridge Regression (L2)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# alpha is the L2 regularization strength (λ)
model = Ridge(alpha=0.1)
model.fit(X, y)          # X shape: (n, 1)

print(f"weight = {model.coef_[0]:.4f}")
print(f"bias   = {model.intercept_[0]:.4f}")
print(f"MSE    = {mean_squared_error(y, model.predict(X)):.4f}")

# Loss = MSE + alpha * sum(w²)
# Larger alpha → more regularization → smaller weights`,

  logistic: (p) => `# scikit-learn — Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(
    solver='lbfgs',
    max_iter=${p.epochs},
    C=1.0          # inverse of regularization strength
)
model.fit(X, y)  # X shape: (n, 2), y: binary labels

print(f"Accuracy: {accuracy_score(y, model.predict(X)):.2%}")
print(f"Weights:  {model.coef_[0]}")
print(f"Bias:     {model.intercept_[0]:.4f}")
# Decision boundary: w1*x1 + w2*x2 + b = 0`,

  knn: (p) => `# scikit-learn — K-Nearest Neighbors (k=${p.k})
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

model = KNeighborsClassifier(
    n_neighbors=${p.k},
    metric='euclidean'
)
model.fit(X_train, y_train)   # No actual "training" — lazy learner

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# Predict a single point:
# label = model.predict([[x1, x2]])[0]`,

  naiveBayes: () => `# scikit-learn — Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# P(y | x1, x2) ∝ P(y) * P(x1|y) * P(x2|y)
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# Class priors: model.class_prior_
# Feature means per class: model.theta_
# Feature variances per class: model.var_`,

  svm: (p) => `# scikit-learn — Support Vector Machine
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model = SVC(
    kernel='linear',
    C=${p.C},          # margin penalty (higher C = less margin)
    max_iter=${p.epochs}
)
model.fit(X_train, y_train)

print(f"Accuracy:       {accuracy_score(y_train, model.predict(X_train)):.2%}")
print(f"Support vectors: {model.n_support_}")
print(f"Weights:         {model.coef_[0]}")
# Boundary: w · x + b = 0`,

  decisionTree: (p) => `# scikit-learn — Decision Tree (CART)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier(
    max_depth=${p.maxDepth},
    criterion='gini'   # split metric: Gini Impurity
)
model.fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_train, model.predict(X_train)):.2%}")
print(export_text(model, feature_names=['x', 'y']))

# Gini Impurity = 1 - Σ pᵢ²
# Feature importances:
# print(model.feature_importances_)`,

  randomForest: (p) => `# scikit-learn — Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(
    n_estimators=${p.nTrees},   # number of trees
    max_depth=${p.maxDepth},
    bootstrap=True,              # random sampling with replacement
    random_state=42
)
model.fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_train, model.predict(X_train)):.2%}")
print(f"Feature importances: {model.feature_importances_}")
# Each tree votes; majority class wins`,

  kmeans: (p) => `# scikit-learn — K-Means Clustering (k=${p.k})
from sklearn.cluster import KMeans
import numpy as np

model = KMeans(
    n_clusters=${p.k},
    init='k-means++',   # smarter centroid initialization
    max_iter=20,
    random_state=42
)
model.fit(X)           # X shape: (n, 2), no labels needed

labels = model.labels_
centroids = model.cluster_centers_
print(f"Inertia (WCSS): {model.inertia_:.4f}")`,

  nn: (p) => `# TensorFlow / Keras — Neural Network MLP
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=(2,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer=keras.optimizers.Adam(${p.learningRate}),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model.fit(X, y, epochs=${p.epochs}, verbose=0)
print(f"Final accuracy: {history.history['accuracy'][-1]:.2%}")`,

  dnn: (p) => `# TensorFlow / Keras — Deep Neural Network
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(2,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer=keras.optimizers.Adam(${p.learningRate}),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model.fit(X, y, epochs=${p.epochs}, verbose=0)
print(f"Final accuracy: {history.history['accuracy'][-1]:.2%}")`,
};

const LANG_LABELS = { js: 'JavaScript', py: 'Python' };

const CodeSnippet = ({ activeTab, params }) => {
  const [lang, setLang] = useState('py');

  const getCode = () => {
    const map = lang === 'py' ? PY_MAP : JS_MAP;
    const fn = map[activeTab];
    return fn ? fn(params) : `# ${activeTab} implementation`;
  };

  return (
    <div className="bg-slate-950 rounded-xl border border-white/5 font-mono text-sm flex flex-col">
      <div className="bg-white/5 px-4 py-2 border-b border-white/5 flex items-center justify-between shrink-0 rounded-t-xl">
        <div className="flex items-center gap-2">
          <Terminal size={14} className="text-brand-500" />
          <span className="text-xs text-slate-400">
            Implementation — {activeTab} &nbsp;·&nbsp; {LANG_LABELS[lang]}
          </span>
        </div>
        <div className="flex items-center bg-slate-900 rounded-lg p-0.5 gap-0.5">
          {['py', 'js'].map(l => (
            <button
              key={l}
              onClick={() => setLang(l)}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                lang === l
                  ? 'bg-brand-500 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {l === 'py' ? 'Python' : 'JS'}
            </button>
          ))}
        </div>
      </div>
      <pre className="p-4 text-slate-300 overflow-x-auto overflow-y-auto text-xs leading-relaxed max-h-44 rounded-b-xl">
        <code>{getCode()}</code>
      </pre>
    </div>
  );
};

export default CodeSnippet;
