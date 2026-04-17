import * as tf from '@tensorflow/tfjs';

// ─────────────────────────────────────────────
//  LINEAR REGRESSION
// ─────────────────────────────────────────────
export const trainLinearRegression = async (data, params, onEpoch) => {
  const { learningRate = 0.1, epochs = 100, validationSplit = 0.2 } = params;
  const xs = tf.tensor2d(data.map(d => [d.x]));
  const ys = tf.tensor2d(data.map(d => [d.y]));
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ optimizer: tf.train.sgd(learningRate), loss: 'meanSquaredError' });

  // Pre-compute for R² (doesn't change across epochs)
  const yMean  = data.reduce((s, d) => s + d.y, 0) / data.length;
  const ssTot  = data.reduce((s, d) => s + (d.y - yMean) ** 2, 0) || 1;

  await model.fit(xs, ys, {
    epochs,
    validationSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) {
          const weight = model.getWeights()[0].dataSync()[0];
          const bias   = model.getWeights()[1].dataSync()[0];
          const ssRes  = data.reduce((s, d) => s + (d.y - (weight * d.x + bias)) ** 2, 0);
          const r2     = Math.max(0, 1 - ssRes / ssTot);
          await onEpoch(epoch, { ...logs, weight, bias, acc: r2 });
        }
        await tf.nextFrame();
      }
    }
  });

  const weight = model.getWeights()[0].dataSync()[0];
  const bias = model.getWeights()[1].dataSync()[0];
  tf.dispose([xs, ys]);
  return { weight, bias, model };
};

// ─────────────────────────────────────────────
//  POLYNOMIAL REGRESSION
// ─────────────────────────────────────────────
export const trainPolynomialRegression = async (data, params, onEpoch) => {
  const { learningRate = 0.01, epochs = 200, degree = 2, validationSplit = 0.2 } = params;

  // Expand features: [x, x^2, ..., x^degree]
  const expandPoly = (x) => Array.from({ length: degree }, (_, i) => Math.pow(x, i + 1));
  const xs = tf.tensor2d(data.map(d => expandPoly(d.x)));
  const ys = tf.tensor2d(data.map(d => [d.y]));

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [degree] }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'meanSquaredError' });

  const xMin  = Math.min(...data.map(d => d.x));
  const xMax  = Math.max(...data.map(d => d.x));

  // Pre-compute for R² (doesn't change across epochs)
  const yMean = data.reduce((s, d) => s + d.y, 0) / data.length;
  const ssTot = data.reduce((s, d) => s + (d.y - yMean) ** 2, 0) || 1;

  await model.fit(xs, ys, {
    epochs,
    validationSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) {
          // Generate curve points for visualization
          const curvePoints = 50;
          const step   = (xMax - xMin) / curvePoints;
          const testXs = Array.from({ length: curvePoints + 1 }, (_, i) => xMin + i * step);
          const expanded = tf.tensor2d(testXs.map(x => expandPoly(x)));
          const preds    = model.predict(expanded).dataSync();
          tf.dispose(expanded);
          const curve = testXs.map((x, i) => ({ x, y: preds[i] }));

          // R² on training data
          const trainExpanded = tf.tensor2d(data.map(d => expandPoly(d.x)));
          const trainPreds    = model.predict(trainExpanded).dataSync();
          tf.dispose(trainExpanded);
          const ssRes = data.reduce((s, d, i) => s + (d.y - trainPreds[i]) ** 2, 0);
          const r2    = Math.max(0, 1 - ssRes / ssTot);

          await onEpoch(epoch, { ...logs, curve, type: 'poly', acc: r2 });
        }
        await tf.nextFrame();
      }
    }
  });

  const curvePoints = 50;
  const step = (xMax - xMin) / curvePoints;
  const testXs = Array.from({ length: curvePoints + 1 }, (_, i) => xMin + i * step);
  const expanded = tf.tensor2d(testXs.map(x => expandPoly(x)));
  const preds = model.predict(expanded).dataSync();
  tf.dispose([xs, ys, expanded]);
  const curve = testXs.map((x, i) => ({ x, y: preds[i] }));
  return { model, curve, type: 'poly' };
};

// ─────────────────────────────────────────────
//  LOGISTIC REGRESSION
// ─────────────────────────────────────────────
export const trainLogisticRegression = async (data, params, onEpoch) => {
  const { learningRate = 0.1, epochs = 100, validationSplit = 0.2 } = params;
  const xs = tf.tensor2d(data.map(d => [d.x, d.y]));
  const ys = tf.tensor2d(data.map(d => [d.label]));
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [2] }));
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  await model.fit(xs, ys, {
    epochs,
    validationSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) {
          const weights = model.getWeights()[0].dataSync();
          const bias = model.getWeights()[1].dataSync()[0];
          await onEpoch(epoch, { ...logs, weights: Array.from(weights), bias, type: 'logistic' });
        }
        await tf.nextFrame();
      }
    }
  });

  const weights = model.getWeights()[0].dataSync();
  const bias = model.getWeights()[1].dataSync()[0];
  tf.dispose([xs, ys]);
  return { model, weights: Array.from(weights), bias, type: 'logistic' };
};

// ─────────────────────────────────────────────
//  RIDGE REGRESSION (L2) via TF.js
// ─────────────────────────────────────────────
export const trainRidgeRegression = async (data, params, onEpoch) => {
  const { learningRate = 0.05, epochs = 150, lambda = 0.1 } = params;
  const xs = tf.tensor2d(data.map(d => [d.x]));
  const ys = tf.tensor2d(data.map(d => [d.y]));

  const w = tf.variable(tf.randomNormal([1, 1]));
  const b = tf.variable(tf.zeros([1]));

  const predict = (x) => x.matMul(w).add(b);
  const optimizer = tf.train.sgd(learningRate);

  // Pre-compute for R²
  const yMean = data.reduce((s, d) => s + d.y, 0) / data.length;
  const ssTot = data.reduce((s, d) => s + (d.y - yMean) ** 2, 0) || 1;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const loss = optimizer.minimize(() => {
      const preds = predict(xs);
      const mse = tf.losses.meanSquaredError(ys, preds);
      const l2 = tf.mul(lambda, tf.sum(tf.square(w)));
      return mse.add(l2);
    }, true);

    if (onEpoch) {
      const weight  = w.dataSync()[0];
      const bias    = b.dataSync()[0];
      const lossVal = loss.dataSync()[0];
      const ssRes   = data.reduce((s, d) => s + (d.y - (weight * d.x + bias)) ** 2, 0);
      const r2      = Math.max(0, 1 - ssRes / ssTot);
      await onEpoch(epoch, { loss: lossVal, weight, bias, acc: r2 });
    }
    loss.dispose();
    await tf.nextFrame();
  }

  const weight = w.dataSync()[0];
  const bias = b.dataSync()[0];
  tf.dispose([xs, ys, w, b]);
  return { weight, bias, type: 'ridge' };
};

// ─────────────────────────────────────────────
//  KNN (Pure JS)
// ─────────────────────────────────────────────
export const predictKNN = (trainData, testPoint, k) => {
  const distances = trainData.map(point => {
    const d = Math.sqrt(
      Math.pow(point.x - testPoint.x, 2) +
      Math.pow(point.y - testPoint.y, 2)
    );
    return { ...point, distance: d };
  });
  const nearest = distances.sort((a, b) => a.distance - b.distance).slice(0, k);
  const counts = nearest.reduce((acc, curr) => {
    acc[curr.label] = (acc[curr.label] || 0) + 1;
    return acc;
  }, {});
  return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
};

// Build full KNN decision surface for visualization
export const buildKNNSurface = (trainData, k = 3, resolution = 30) => {
  const xVals = trainData.map(d => d.x);
  const yVals = trainData.map(d => d.y);
  const xMin = Math.min(...xVals) - 1, xMax = Math.max(...xVals) + 1;
  const yMin = Math.min(...yVals) - 1, yMax = Math.max(...yVals) + 1;

  const cells = [];
  for (let i = 0; i <= resolution; i++) {
    for (let j = 0; j <= resolution; j++) {
      const x = xMin + (i / resolution) * (xMax - xMin);
      const y = yMin + (j / resolution) * (yMax - yMin);
      const label = parseInt(predictKNN(trainData, { x, y }, k));
      cells.push({ x, y, label });
    }
  }
  return { cells, xMin, xMax, yMin, yMax };
};

// ─────────────────────────────────────────────
//  K-MEANS CLUSTERING
// ─────────────────────────────────────────────
export const trainKMeans = (data, params, onStep) => {
  const { k = 3, maxIterations = 20 } = params;
  let centroids = [...data].sort(() => 0.5 - Math.random()).slice(0, k).map(p => ({ x: p.x, y: p.y }));
  let assignments = new Array(data.length).fill(-1);
  let changed = true;
  let iteration = 0;

  while (changed && iteration < maxIterations) {
    changed = false;
    iteration++;

    data.forEach((point, i) => {
      let minDist = Infinity;
      let closest = -1;
      centroids.forEach((centroid, j) => {
        const d = Math.sqrt(Math.pow(point.x - centroid.x, 2) + Math.pow(point.y - centroid.y, 2));
        if (d < minDist) { minDist = d; closest = j; }
      });
      if (assignments[i] !== closest) { assignments[i] = closest; changed = true; }
    });

    const newCentroids = centroids.map((_, j) => {
      const pts = data.filter((_, i) => assignments[i] === j);
      if (pts.length === 0) return centroids[j];
      return { x: pts.reduce((s, p) => s + p.x, 0) / pts.length, y: pts.reduce((s, p) => s + p.y, 0) / pts.length };
    });
    centroids = newCentroids;

    if (onStep) onStep(iteration, { centroids: [...centroids], assignments: [...assignments] });
  }

  const clusteredData = data.map((p, i) => ({ ...p, label: assignments[i] }));
  return { centroids, clusteredData, assignments };
};

// ─────────────────────────────────────────────
//  NAIVE BAYES (Gaussian, Pure JS)
// ─────────────────────────────────────────────
export const trainNaiveBayes = (data, onStep) => {
  const classes = [...new Set(data.map(d => d.label))];
  const stats = {};

  classes.forEach(cls => {
    const pts = data.filter(d => d.label === cls);
    const xMean = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const yMean = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    const xVar = pts.reduce((s, p) => s + Math.pow(p.x - xMean, 2), 0) / pts.length || 0.001;
    const yVar = pts.reduce((s, p) => s + Math.pow(p.y - yMean, 2), 0) / pts.length || 0.001;
    stats[cls] = { mean: { x: xMean, y: yMean }, variance: { x: xVar, y: yVar }, prior: pts.length / data.length };
    if (onStep) onStep(cls, stats[cls]);
  });

  const gaussianPdf = (x, mean, variance) => {
    return (1 / Math.sqrt(2 * Math.PI * variance)) * Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
  };

  const predict = (x, y) => {
    let maxProb = -Infinity;
    let bestClass = classes[0];
    classes.forEach(cls => {
      const { mean, variance, prior } = stats[cls];
      const prob = Math.log(prior) + Math.log(gaussianPdf(x, mean.x, variance.x)) + Math.log(gaussianPdf(y, mean.y, variance.y));
      if (prob > maxProb) { maxProb = prob; bestClass = cls; }
    });
    return parseInt(bestClass);
  };

  // Build surface
  const xVals = data.map(d => d.x), yVals = data.map(d => d.y);
  const xMin = Math.min(...xVals) - 1, xMax = Math.max(...xVals) + 1;
  const yMin = Math.min(...yVals) - 1, yMax = Math.max(...yVals) + 1;
  const resolution = 30;
  const cells = [];
  for (let i = 0; i <= resolution; i++) {
    for (let j = 0; j <= resolution; j++) {
      const x = xMin + (i / resolution) * (xMax - xMin);
      const y = yMin + (j / resolution) * (yMax - yMin);
      cells.push({ x, y, label: predict(x, y) });
    }
  }

  const nbCorrect = data.filter(d => predict(d.x, d.y) === d.label).length;
  return { stats, predict, type: 'naiveBayes', cells, xMin, xMax, yMin, yMax, accuracy: nbCorrect / data.length };
};

// ─────────────────────────────────────────────
//  DECISION TREE (simple CART, Pure JS)
// ─────────────────────────────────────────────
function gini(groups, classes) {
  const nSamples = groups.reduce((s, g) => s + g.length, 0);
  let score = 0;
  groups.forEach(group => {
    const size = group.length;
    if (size === 0) return;
    let p = 0;
    classes.forEach(cls => { const c = group.filter(r => r.label === cls).length / size; p += c * c; });
    score += (1 - p) * (size / nSamples);
  });
  return score;
}

function splitData(data, feature, threshold) {
  const left = data.filter(d => d[feature] <= threshold);
  const right = data.filter(d => d[feature] > threshold);
  return { left, right };
}

function bestSplit(data) {
  const classes = [...new Set(data.map(d => d.label))];
  let bestGini = Infinity, bestFeature = null, bestThreshold = null, bestGroups = null;
  ['x', 'y'].forEach(feature => {
    const values = [...new Set(data.map(d => d[feature]))].sort((a, b) => a - b);
    values.forEach(threshold => {
      const { left, right } = splitData(data, feature, threshold);
      const g = gini([left, right], classes);
      if (g < bestGini) { bestGini = g; bestFeature = feature; bestThreshold = threshold; bestGroups = { left, right }; }
    });
  });
  return { feature: bestFeature, threshold: bestThreshold, groups: bestGroups, gini: bestGini };
}

function toLeaf(group) {
  const counts = {};
  group.forEach(d => { counts[d.label] = (counts[d.label] || 0) + 1; });
  return { leaf: true, label: parseInt(Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b)) };
}

function buildTree(data, depth = 0, maxDepth = 4, minSize = 2, importances = null) {
  const classes = [...new Set(data.map(d => d.label))];
  if (classes.length === 1 || data.length <= minSize || depth >= maxDepth) return toLeaf(data);
  const { feature, threshold, groups, gini: splitGini } = bestSplit(data);
  if (!feature || !groups.left.length || !groups.right.length) return toLeaf(data);

  // Accumulate weighted gini gain per feature for importance
  if (importances) {
    const n     = data.length;
    const nL    = groups.left.length;
    const nR    = groups.right.length;
    const giniParent = gini([data], classes) / (n === 0 ? 1 : 1); // already weighted
    // Simpler: weighted impurity reduction
    const gain  = giniParent - (nL / n) * splitGini - (nR / n) * splitGini;
    importances[feature] = (importances[feature] || 0) + gain * n;
  }

  return {
    feature, threshold,
    left:  buildTree(groups.left,  depth + 1, maxDepth, minSize, importances),
    right: buildTree(groups.right, depth + 1, maxDepth, minSize, importances),
  };
}

function predictTree(node, row) {
  if (node.leaf) return node.label;
  if (row[node.feature] <= node.threshold) return predictTree(node.left, row);
  return predictTree(node.right, row);
}

export const trainDecisionTree = (data, params, onStep) => {
  const { maxDepth = 4 } = params;
  const rawImportances = {};
  const tree = buildTree(data, 0, maxDepth, 2, rawImportances);

  // Normalize importances so they sum to 1
  const totalImp = Object.values(rawImportances).reduce((s, v) => s + v, 0) || 1;
  const featureImportance = {
    x: (rawImportances.x || 0) / totalImp,
    y: (rawImportances.y || 0) / totalImp,
  };

  if (onStep) onStep(1, { tree, type: 'decisionTree' });

  // Build surface
  const xVals = data.map(d => d.x), yVals = data.map(d => d.y);
  const xMin = Math.min(...xVals) - 1, xMax = Math.max(...xVals) + 1;
  const yMin = Math.min(...yVals) - 1, yMax = Math.max(...yVals) + 1;
  const resolution = 40;
  const cells = [];
  for (let i = 0; i <= resolution; i++) {
    for (let j = 0; j <= resolution; j++) {
      const x = xMin + (i / resolution) * (xMax - xMin);
      const y = yMin + (j / resolution) * (yMax - yMin);
      cells.push({ x, y, label: predictTree(tree, { x, y }) });
    }
  }
  const dtCorrect = data.filter(d => predictTree(tree, d) === d.label).length;
  return { tree, cells, xMin, xMax, yMin, yMax, type: 'decisionTree', accuracy: dtCorrect / data.length, featureImportance };
};

// ─────────────────────────────────────────────
//  RANDOM FOREST (ensemble of decision trees)
// ─────────────────────────────────────────────
export const trainRandomForest = (data, params, onStep) => {
  const { nTrees = 10, maxDepth = 3 } = params;
  const trees = [];
  const aggregateImportances = { x: 0, y: 0 };

  for (let t = 0; t < nTrees; t++) {
    // Bootstrap sample
    const sample = Array.from({ length: data.length }, () => data[Math.floor(Math.random() * data.length)]);
    const rawImp = {};
    const tree = buildTree(sample, 0, maxDepth, 2, rawImp);
    trees.push(tree);
    // Accumulate importances across trees
    const total = Object.values(rawImp).reduce((s, v) => s + v, 0) || 1;
    aggregateImportances.x += (rawImp.x || 0) / total;
    aggregateImportances.y += (rawImp.y || 0) / total;
    if (onStep) onStep(t + 1, { treesBuilt: t + 1, total: nTrees });
  }

  // Average across trees
  const featureImportance = {
    x: aggregateImportances.x / nTrees,
    y: aggregateImportances.y / nTrees,
  };

  const predict = (x, y) => {
    const votes = {};
    trees.forEach(tree => {
      const pred = String(predictTree(tree, { x, y }));
      votes[pred] = (votes[pred] || 0) + 1;
    });
    return parseInt(Object.keys(votes).reduce((a, b) => votes[a] > votes[b] ? a : b));
  };

  // Build surface
  const xVals = data.map(d => d.x), yVals = data.map(d => d.y);
  const xMin = Math.min(...xVals) - 1, xMax = Math.max(...xVals) + 1;
  const yMin = Math.min(...yVals) - 1, yMax = Math.max(...yVals) + 1;
  const resolution = 35;
  const cells = [];
  for (let i = 0; i <= resolution; i++) {
    for (let j = 0; j <= resolution; j++) {
      const x = xMin + (i / resolution) * (xMax - xMin);
      const y = yMin + (j / resolution) * (yMax - yMin);
      cells.push({ x, y, label: predict(x, y) });
    }
  }
  const rfCorrect = data.filter(d => predict(d.x, d.y) === d.label).length;
  return { trees, cells, xMin, xMax, yMin, yMax, type: 'randomForest', nTrees, accuracy: rfCorrect / data.length, featureImportance };
};

// ─────────────────────────────────────────────
//  SVM LINEAR (gradient descent approximation, Pure JS)
// ─────────────────────────────────────────────
export const trainSVM = async (data, params, onEpoch) => {
  const { learningRate = 0.001, epochs = 200, C = 1.0 } = params;
  // Labels must be -1 or 1
  const labeled = data.map(d => ({ ...d, svmLabel: d.label === 0 ? -1 : 1 }));

  let w = [0, 0];
  let b = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    labeled.forEach(d => {
      const margin = d.svmLabel * (w[0] * d.x + w[1] * d.y + b);
      if (margin < 1) {
        w[0] = w[0] - learningRate * (2 * (1 / epoch + 1) * w[0] - C * d.svmLabel * d.x);
        w[1] = w[1] - learningRate * (2 * (1 / epoch + 1) * w[1] - C * d.svmLabel * d.y);
        b = b + learningRate * C * d.svmLabel;
      } else {
        w[0] = w[0] - learningRate * (2 * (1 / epoch + 1) * w[0]);
        w[1] = w[1] - learningRate * (2 * (1 / epoch + 1) * w[1]);
      }
    });

    if (onEpoch && (epoch % 5 === 0 || epoch === epochs - 1)) {
      const loss = labeled.reduce((s, d) => {
        const margin = d.svmLabel * (w[0] * d.x + w[1] * d.y + b);
        return s + Math.max(0, 1 - margin);
      }, 0) / labeled.length;
      const correct = labeled.filter(d => (w[0] * d.x + w[1] * d.y + b >= 0 ? 1 : -1) === d.svmLabel).length;
      const acc = correct / labeled.length;
      await onEpoch(epoch, { weights: [...w], bias: b, loss, acc, type: 'svm' });
    }
    await tf.nextFrame();
  }

  return { weights: w, bias: b, type: 'svm' };
};

// ─────────────────────────────────────────────
//  BACKPROPAGATION (explicit 3-layer MLP + gradient norms)
// ─────────────────────────────────────────────
export const trainBackprop = async (data, params, onEpoch) => {
  const { learningRate = 0.05, epochs = 150, validationSplit = 0.2 } = params;

  const allXs = tf.tensor2d(data.map(d => [d.x, d.y]));
  const allYs = tf.tensor2d(data.map(d => [d.label]));

  // Architecture: 2 → 8 (relu) → 4 (relu) → 1 (sigmoid)
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [2] }));
  model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  // Use vanilla SGD so the "raw" gradient descent is visible
  model.compile({ optimizer: tf.train.sgd(learningRate), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  await model.fit(allXs, allYs, {
    epochs,
    validationSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) {
          // ── Gradient norms per layer (W only, skip bias) ──────────────
          let gradNorms = [];
          try {
            const { grads } = tf.variableGrads(() => {
              const pred = model.predict(allXs);
              return tf.losses.logLoss(allYs, pred).mean();
            });
            model.layers.forEach((layer, li) => {
              const kv = layer.trainableWeights.find(v => v.name.includes('kernel'));
              if (kv && grads[kv.name]) {
                gradNorms.push({ layer: `Layer ${li + 1}`, norm: grads[kv.name].norm().dataSync()[0] });
              }
            });
            Object.values(grads).forEach(g => { try { g.dispose(); } catch (_) {} });
          } catch (_) { gradNorms = []; }

          // ── Decision boundary surface ──────────────────────────────────
          const resolution = 25;
          const xVals = data.map(d => d.x), yVals = data.map(d => d.y);
          const xMin = Math.min(...xVals) - 1, xMax = Math.max(...xVals) + 1;
          const yMin = Math.min(...yVals) - 1, yMax = Math.max(...yVals) + 1;
          const testPoints = [];
          for (let i = 0; i <= resolution; i++)
            for (let j = 0; j <= resolution; j++)
              testPoints.push([xMin + (i / resolution) * (xMax - xMin), yMin + (j / resolution) * (yMax - yMin)]);
          const testTensor = tf.tensor2d(testPoints);
          const preds = model.predict(testTensor).dataSync();
          tf.dispose(testTensor);
          const cells = testPoints.map(([x, y], idx) => ({ x, y, prob: preds[idx] }));
          const weights = model.layers.map(layer => layer.getWeights()[0]?.arraySync() || []);

          await onEpoch(epoch, { ...logs, weights, cells, xMin, xMax, yMin, yMax, type: 'backprop', gradNorms });
        }
        await tf.nextFrame();
      }
    }
  });

  const weights = model.layers.map(layer => layer.getWeights()[0]?.arraySync() || []);
  tf.dispose([allXs, allYs]);
  return { model, weights, type: 'backprop' };
};

// ─────────────────────────────────────────────
//  FEEDFORWARD NEURAL NETWORK (MLP)
// ─────────────────────────────────────────────
export const trainFFNN = async (data, params, config, onEpoch) => {
  const { learningRate = 0.01, epochs = 100, validationSplit = 0.2 } = params;
  const { hiddenLayers = [4, 4], activation = 'relu' } = config;

  const xs = tf.tensor2d(data.map(d => [d.x, d.y]));
  const ys = tf.tensor2d(data.map(d => [d.label]));

  const model = tf.sequential();
  hiddenLayers.forEach((units, i) => {
    model.add(tf.layers.dense({ units, activation, inputShape: i === 0 ? [2] : undefined }));
  });
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  await model.fit(xs, ys, {
    epochs,
    validationSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) {
          // Decision boundary surface for NN
          const resolution = 25;
          const xVals = data.map(d => d.x), yVals = data.map(d => d.y);
          const xMin = Math.min(...xVals) - 1, xMax = Math.max(...xVals) + 1;
          const yMin = Math.min(...yVals) - 1, yMax = Math.max(...yVals) + 1;

          const testPoints = [];
          for (let i = 0; i <= resolution; i++) {
            for (let j = 0; j <= resolution; j++) {
              testPoints.push([ xMin + (i / resolution) * (xMax - xMin), yMin + (j / resolution) * (yMax - yMin) ]);
            }
          }
          const testTensor = tf.tensor2d(testPoints);
          const predictions = model.predict(testTensor).dataSync();
          tf.dispose(testTensor);
          const cells = testPoints.map(([x, y], idx) => ({ x, y, prob: predictions[idx] }));
          const weights = model.layers.map(layer => layer.getWeights()[0]?.arraySync() || []);
          await onEpoch(epoch, { ...logs, weights, cells, xMin, xMax, yMin, yMax, type: 'nn' });
        }
        await tf.nextFrame();
      }
    }
  });

  const weights = model.layers.map(layer => layer.getWeights()[0]?.arraySync() || []);
  tf.dispose([xs, ys]);
  return { model, weights, type: 'nn' };
};

// ─────────────────────────────────────────────
//  PCA (Principal Component Analysis, Pure JS)
// ─────────────────────────────────────────────
export const trainPCA = (data) => {
  if (!data || data.length < 3) return null;

  const n  = data.length;
  const mx = data.reduce((s, d) => s + d.x, 0) / n;
  const my = data.reduce((s, d) => s + d.y, 0) / n;

  // Covariance matrix elements
  const cxx = data.reduce((s, d) => s + (d.x - mx) ** 2, 0) / (n - 1);
  const cyy = data.reduce((s, d) => s + (d.y - my) ** 2, 0) / (n - 1);
  const cxy = data.reduce((s, d) => s + (d.x - mx) * (d.y - my), 0) / (n - 1);

  // Analytical eigenvalues for 2×2 symmetric matrix
  const tr   = cxx + cyy;
  const det  = cxx * cyy - cxy * cxy;
  const disc = Math.sqrt(Math.max(0, (tr * 0.5) ** 2 - det));
  const lam1 = tr * 0.5 + disc;   // larger eigenvalue → PC1
  const lam2 = tr * 0.5 - disc;   // smaller eigenvalue → PC2

  // Eigenvector for PC1
  let v1x, v1y;
  if (Math.abs(cxy) > 1e-10) {
    v1x = lam1 - cyy; v1y = cxy;
  } else {
    v1x = cxx >= cyy ? 1 : 0; v1y = cxx >= cyy ? 0 : 1;
  }
  const norm1 = Math.sqrt(v1x ** 2 + v1y ** 2) || 1;
  v1x /= norm1; v1y /= norm1;
  const v2x = -v1y, v2y = v1x; // PC2 is orthogonal

  const totalVar     = lam1 + lam2 || 1;
  const explainedV1  = lam1 / totalVar;
  const explainedV2  = lam2 / totalVar;

  // Project data onto PCs (for coloring)
  const projections = data.map((d, i) => {
    const cx = d.x - mx, cy = d.y - my;
    return { ...d, idx: i, pc1: cx * v1x + cy * v1y, pc2: cx * v2x + cy * v2y };
  });
  const pc1vals = projections.map(p => p.pc1);
  const pc1Min  = Math.min(...pc1vals);
  const pc1Max  = Math.max(...pc1vals) || 1;

  // Arrow lengths proportional to sqrt(eigenvalue)
  const scale = 2.5;
  const len1  = Math.sqrt(Math.max(lam1, 0)) * scale;
  const len2  = Math.sqrt(Math.max(lam2, 0)) * scale;

  return {
    type:      'pca',
    centroid:  { x: mx, y: my },
    pc1:       { dx: v1x * len1, dy: v1y * len1, eigenvalue: lam1, explained: explainedV1 },
    pc2:       { dx: v2x * len2, dy: v2y * len2, eigenvalue: lam2, explained: explainedV2 },
    projections,
    pc1Min, pc1Max,
    explainedVar1: explainedV1,
    explainedVar2: explainedV2,
    accuracy: explainedV1,  // reuse liveMetrics.acc to show PC1 explained variance
  };
};

// ─────────────────────────────────────────────
//  HELPER: Predict all sequences and compute accuracy
// ─────────────────────────────────────────────
async function predictSequences(model, data) {
  const seqLen = data[0].seq.length;
  const xs = tf.tensor3d(data.map(d => d.seq.map(v => [v]))); // [n, seqLen, 1]
  const probs = model.predict(xs).dataSync();
  tf.dispose(xs);
  const predictions = Array.from(probs).map((prob, i) => ({
    prob,
    label: prob >= 0.5 ? 1 : 0,
    correct: (prob >= 0.5 ? 1 : 0) === data[i].label,
  }));
  const accuracy = predictions.filter(p => p.correct).length / predictions.length;
  return { predictions, accuracy };
}

// ─────────────────────────────────────────────
//  CNN (1-D Convolutional Neural Network)
// ─────────────────────────────────────────────
export const trainCNN = async (data, params, onEpoch) => {
  const { learningRate = 0.01, epochs = 100, validationSplit = 0.2 } = params;
  const seqLen = data[0]?.seq?.length || 16;

  // [n, seqLen, 1]
  const xs = tf.tensor3d(data.map(d => d.seq.map(v => [v])));
  const ys = tf.tensor2d(data.map(d => [d.label]));

  const model = tf.sequential();
  model.add(tf.layers.conv1d({ filters: 16, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [seqLen, 1] }));
  model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
  model.add(tf.layers.conv1d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
  model.add(tf.layers.globalMaxPooling1d());
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  await model.fit(xs, ys, {
    epochs,
    validationSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) await onEpoch(epoch, { ...logs, type: 'cnn' });
        await tf.nextFrame();
      }
    }
  });

  const { predictions, accuracy } = await predictSequences(model, data);
  tf.dispose([xs, ys]);
  return { model, type: 'cnn', seqLen, predictions, accuracy };
};

// ─────────────────────────────────────────────
//  RNN (Simple Recurrent Neural Network)
// ─────────────────────────────────────────────
export const trainRNN = async (data, params, onEpoch) => {
  const { learningRate = 0.01, epochs = 100, validationSplit = 0.2 } = params;
  const seqLen = data[0]?.seq?.length || 16;

  const xs = tf.tensor3d(data.map(d => d.seq.map(v => [v])));
  const ys = tf.tensor2d(data.map(d => [d.label]));

  const model = tf.sequential();
  model.add(tf.layers.simpleRNN({ units: 32, returnSequences: false, inputShape: [seqLen, 1] }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  await model.fit(xs, ys, {
    epochs,
    validationSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) await onEpoch(epoch, { ...logs, type: 'rnn' });
        await tf.nextFrame();
      }
    }
  });

  const { predictions, accuracy } = await predictSequences(model, data);
  tf.dispose([xs, ys]);
  return { model, type: 'rnn', seqLen, predictions, accuracy };
};

// ─────────────────────────────────────────────
//  LSTM (Long Short-Term Memory)
// ─────────────────────────────────────────────
export const trainLSTM = async (data, params, onEpoch) => {
  const { learningRate = 0.01, epochs = 100, validationSplit = 0.2 } = params;
  const seqLen = data[0]?.seq?.length || 16;

  const xs = tf.tensor3d(data.map(d => d.seq.map(v => [v])));
  const ys = tf.tensor2d(data.map(d => [d.label]));

  const model = tf.sequential();
  model.add(tf.layers.lstm({ units: 32, returnSequences: false, inputShape: [seqLen, 1] }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  await model.fit(xs, ys, {
    epochs,
    validationSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) await onEpoch(epoch, { ...logs, type: 'lstm' });
        await tf.nextFrame();
      }
    }
  });

  const { predictions, accuracy } = await predictSequences(model, data);
  tf.dispose([xs, ys]);
  return { model, type: 'lstm', seqLen, predictions, accuracy };
};

// ─────────────────────────────────────────────
//  TRANSFORMER (Scaled Dot-Product Self-Attention)
// ─────────────────────────────────────────────
class SelfAttentionLayer extends tf.layers.Layer {
  constructor(config) {
    super({ name: 'selfAttention', ...config });
    this.dModel = (config && config.dModel) || 8;
  }
  build(inputShape) {
    const inDim = inputShape[inputShape.length - 1];
    const d = this.dModel;
    const init = tf.initializers.glorotNormal({});
    this.Wq = this.addWeight('Wq', [inDim, d], 'float32', init);
    this.Wk = this.addWeight('Wk', [inDim, d], 'float32', init);
    this.Wv = this.addWeight('Wv', [inDim, d], 'float32', init);
    this.built = true;
  }
  call(inputs) {
    const inp = Array.isArray(inputs) ? inputs[0] : inputs;
    const q = tf.matMul(inp, this.Wq.read());          // [B, T, d]
    const k = tf.matMul(inp, this.Wk.read());
    const v = tf.matMul(inp, this.Wv.read());
    const scale = tf.scalar(Math.sqrt(this.dModel));
    const scores = tf.matMul(q, k, false, true).div(scale); // [B, T, T]
    const weights = tf.softmax(scores, -1);
    return tf.matMul(weights, v);                      // [B, T, d]
  }
  computeOutputShape(inputShape) {
    return [...inputShape.slice(0, -1), this.dModel];
  }
  getConfig() { return { ...super.getConfig(), dModel: this.dModel }; }
  static get className() { return 'SelfAttentionLayer'; }
}
tf.serialization.registerClass(SelfAttentionLayer);

export const trainTransformer = async (data, params, onEpoch) => {
  const { learningRate = 0.01, epochs = 100, validationSplit = 0.2 } = params;
  const seqLen = data[0]?.seq?.length || 16;

  const xs = tf.tensor3d(data.map(d => d.seq.map(v => [v]))); // [n, seqLen, 1]
  const ys = tf.tensor2d(data.map(d => [d.label]));

  const model = tf.sequential();
  // Project 1 input feature → 8-dim embedding for each token
  model.add(tf.layers.dense({ units: 8, inputShape: [seqLen, 1] }));  // [B, seqLen, 8]
  // Self-attention over all tokens
  model.add(new SelfAttentionLayer({ dModel: 8 }));                    // [B, seqLen, 8]
  // Aggregate tokens → fixed-size vector
  model.add(tf.layers.globalAveragePooling1d());                       // [B, 8]
  // Feed-forward head
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  await model.fit(xs, ys, {
    epochs,
    validationSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) await onEpoch(epoch, { ...logs, type: 'transformer' });
        await tf.nextFrame();
      }
    }
  });

  const { predictions, accuracy } = await predictSequences(model, data);
  tf.dispose([xs, ys]);
  return { model, type: 'transformer', seqLen, predictions, accuracy };
};

// ─────────────────────────────────────────────
//  EXPORT MODEL
// ─────────────────────────────────────────────
export const exportModel = async (model, name = 'intellilearn-model') => {
  if (!model) return;
  await model.save(`downloads://${name}`);
};

// ─────────────────────────────────────────────
//  LOGIC GATE DATASETS
// ─────────────────────────────────────────────
export const LOGIC_GATES = {
  AND: [
    { x: 0, y: 0, label: 0 }, { x: 0, y: 1, label: 0 },
    { x: 1, y: 0, label: 0 }, { x: 1, y: 1, label: 1 }
  ],
  OR: [
    { x: 0, y: 0, label: 0 }, { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 }, { x: 1, y: 1, label: 1 }
  ],
  XOR: [
    { x: 0, y: 0, label: 0 }, { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 }, { x: 1, y: 1, label: 0 }
  ]
};
