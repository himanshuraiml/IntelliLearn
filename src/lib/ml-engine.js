import * as tf from '@tensorflow/tfjs';

// ─────────────────────────────────────────────
//  LINEAR REGRESSION
// ─────────────────────────────────────────────
export const trainLinearRegression = async (data, params, onEpoch) => {
  const { learningRate = 0.1, epochs = 100 } = params;
  const xs = tf.tensor2d(data.map(d => [d.x]));
  const ys = tf.tensor2d(data.map(d => [d.y]));
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ optimizer: tf.train.sgd(learningRate), loss: 'meanSquaredError' });

  await model.fit(xs, ys, {
    epochs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) {
          const weight = model.getWeights()[0].dataSync()[0];
          const bias = model.getWeights()[1].dataSync()[0];
          onEpoch(epoch, { ...logs, weight, bias });
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
  const { learningRate = 0.01, epochs = 200, degree = 2 } = params;

  // Expand features: [x, x^2, ..., x^degree]
  const expandPoly = (x) => Array.from({ length: degree }, (_, i) => Math.pow(x, i + 1));
  const xs = tf.tensor2d(data.map(d => expandPoly(d.x)));
  const ys = tf.tensor2d(data.map(d => [d.y]));

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [degree] }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'meanSquaredError' });

  const xMin = Math.min(...data.map(d => d.x));
  const xMax = Math.max(...data.map(d => d.x));

  await model.fit(xs, ys, {
    epochs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) {
          // Generate curve points for visualization
          const curvePoints = 50;
          const step = (xMax - xMin) / curvePoints;
          const testXs = Array.from({ length: curvePoints + 1 }, (_, i) => xMin + i * step);
          const expanded = tf.tensor2d(testXs.map(x => expandPoly(x)));
          const preds = model.predict(expanded).dataSync();
          tf.dispose(expanded);
          const curve = testXs.map((x, i) => ({ x, y: preds[i] }));
          onEpoch(epoch, { ...logs, curve, type: 'poly' });
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
  const { learningRate = 0.1, epochs = 100 } = params;
  const xs = tf.tensor2d(data.map(d => [d.x, d.y]));
  const ys = tf.tensor2d(data.map(d => [d.label]));
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [2] }));
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'binaryCrossentropy' });

  await model.fit(xs, ys, {
    epochs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch) {
          const weights = model.getWeights()[0].dataSync();
          const bias = model.getWeights()[1].dataSync()[0];
          onEpoch(epoch, { ...logs, weights: Array.from(weights), bias, type: 'logistic' });
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

  for (let epoch = 0; epoch < epochs; epoch++) {
    const loss = optimizer.minimize(() => {
      const preds = predict(xs);
      const mse = tf.losses.meanSquaredError(ys, preds);
      const l2 = tf.mul(lambda, tf.sum(tf.square(w)));
      return mse.add(l2);
    }, true);

    if (onEpoch) {
      const weight = w.dataSync()[0];
      const bias = b.dataSync()[0];
      const lossVal = loss.dataSync()[0];
      onEpoch(epoch, { loss: lossVal, weight, bias });
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

  return { stats, predict, type: 'naiveBayes', cells, xMin, xMax, yMin, yMax };
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

function buildTree(data, depth = 0, maxDepth = 4, minSize = 2) {
  const classes = [...new Set(data.map(d => d.label))];
  if (classes.length === 1 || data.length <= minSize || depth >= maxDepth) return toLeaf(data);
  const { feature, threshold, groups } = bestSplit(data);
  if (!feature || !groups.left.length || !groups.right.length) return toLeaf(data);
  return {
    feature, threshold,
    left: buildTree(groups.left, depth + 1, maxDepth, minSize),
    right: buildTree(groups.right, depth + 1, maxDepth, minSize)
  };
}

function predictTree(node, row) {
  if (node.leaf) return node.label;
  if (row[node.feature] <= node.threshold) return predictTree(node.left, row);
  return predictTree(node.right, row);
}

export const trainDecisionTree = (data, params, onStep) => {
  const { maxDepth = 4 } = params;
  const tree = buildTree(data, 0, maxDepth);

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
  return { tree, cells, xMin, xMax, yMin, yMax, type: 'decisionTree' };
};

// ─────────────────────────────────────────────
//  RANDOM FOREST (ensemble of decision trees)
// ─────────────────────────────────────────────
export const trainRandomForest = (data, params, onStep) => {
  const { nTrees = 10, maxDepth = 3 } = params;
  const trees = [];

  for (let t = 0; t < nTrees; t++) {
    // Bootstrap sample
    const sample = Array.from({ length: data.length }, () => data[Math.floor(Math.random() * data.length)]);
    const tree = buildTree(sample, 0, maxDepth);
    trees.push(tree);
    if (onStep) onStep(t + 1, { treesBuilt: t + 1, total: nTrees });
  }

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
  return { trees, cells, xMin, xMax, yMin, yMax, type: 'randomForest', nTrees };
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
      onEpoch(epoch, { weights: [...w], bias: b, loss, type: 'svm' });
    }
    await tf.nextFrame();
  }

  return { weights: w, bias: b, type: 'svm' };
};

// ─────────────────────────────────────────────
//  FEEDFORWARD NEURAL NETWORK (MLP)
// ─────────────────────────────────────────────
export const trainFFNN = async (data, params, config, onEpoch) => {
  const { learningRate = 0.01, epochs = 100 } = params;
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
          onEpoch(epoch, { ...logs, weights, cells, xMin, xMax, yMin, yMax, type: 'nn' });
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
