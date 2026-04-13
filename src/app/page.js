"use client";
import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Play, RotateCcw, Box, Activity, Settings2, BookOpen,
  Download, Share2, Database, ChevronDown,
  Layers, GitBranch, Cpu, Zap, Target, Grid3x3, Pause, Columns2, Eye
} from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import CodeSnippet from '@/components/CodeSnippet';
import NeuralNetworkGraph from '@/components/visualizers/NeuralNetworkGraph';
import ScatterPlot from '@/components/visualizers/ScatterPlot';
import ParameterControl from '@/components/ParameterControl';
import Guide from '@/components/Guide';
import Quiz from '@/components/Quiz';
import TrainingChart from '@/components/TrainingChart';
import ConfusionMatrix from '@/components/ConfusionMatrix';
import FeatureImportance from '@/components/FeatureImportance';
import AlgoExplainer from '@/components/AlgoExplainer';
import {
  trainLinearRegression, trainPolynomialRegression,
  trainLogisticRegression, trainRidgeRegression,
  trainSVM, trainFFNN,
  predictKNN, buildKNNSurface,
  trainKMeans, trainNaiveBayes, trainDecisionTree, trainRandomForest,
  exportModel
} from '@/lib/ml-engine';
import {
  generateLinearData, generatePolynomialData,
  generateClassificationData, generateMultiClassData,
  generateLinearlySeperableData, generateSpiralData,
  generateMoonData, generateIrisData,
  generateClusteringData, generateClusteringData5
} from '@/lib/datasets';
import { saveProgress } from '@/lib/persistence';

// ─── Algorithm Registry ────────────────────────────────────────────────────
const ALGORITHMS = {
  "Regression": [
    { id: 'linear',       label: 'Linear Regression',      icon: <Activity size={15} />,   desc: 'Fit a straight line to predict continuous values.' },
    { id: 'poly',         label: 'Polynomial Regression',  icon: <Zap size={15} />,        desc: 'Fit a curved polynomial to capture non-linear trends.' },
    { id: 'ridge',        label: 'Ridge Regression',       icon: <Target size={15} />,     desc: 'Linear regression with L2 regularization to prevent overfitting.' },
  ],
  "Classification": [
    { id: 'logistic',     label: 'Logistic Regression',    icon: <Activity size={15} />,   desc: 'Binary classification using a sigmoid decision boundary.' },
    { id: 'knn',          label: 'K-Nearest Neighbors',    icon: <BookOpen size={15} />,   desc: 'Classify points by majority vote of K nearest neighbors.' },
    { id: 'naiveBayes',   label: 'Naive Bayes',            icon: <Grid3x3 size={15} />,    desc: "Probabilistic classifier based on Bayes' theorem." },
    { id: 'svm',          label: 'Support Vector Machine', icon: <Target size={15} />,     desc: 'Find the maximum-margin hyperplane that separates classes.' },
    { id: 'decisionTree', label: 'Decision Tree',          icon: <GitBranch size={15} />,  desc: 'Recursively partition data using feature thresholds.' },
    { id: 'randomForest', label: 'Random Forest',          icon: <Layers size={15} />,     desc: 'Ensemble of decision trees that vote on the final class.' },
  ],
  "Clustering": [
    { id: 'kmeans',       label: 'K-Means Clustering',     icon: <Activity size={15} />,   desc: 'Partition unlabeled data into K groups by centroid proximity.' },
  ],
  "Deep Learning": [
    { id: 'nn',           label: 'Neural Network (MLP)',   icon: <Box size={15} />,        desc: 'Multi-layer perceptron that learns non-linear decision boundaries.' },
    { id: 'dnn',          label: 'Deep Neural Net',        icon: <Cpu size={15} />,        desc: 'Deep MLP with more hidden layers for complex patterns.' },
  ],
};

const ALL_ALGOS        = Object.values(ALGORITHMS).flat();
const REGRESSION_ALGOS = new Set(['linear', 'poly', 'ridge']);
const CLUSTERING_ALGOS = new Set(['kmeans']);
const NEEDS_EPOCH      = new Set(['linear', 'poly', 'ridge', 'logistic', 'svm', 'nn', 'dnn']);

// ─── Hyperparameter hints & warnings ──────────────────────────────────────
const PARAM_HINTS = {
  learningRate: 'Step size for weight updates. Too high → overshoot. Too low → slow convergence.',
  epochs:       'Full passes through the dataset. More = longer training, risk of overfitting.',
  k_knn:        'Neighbors to vote. Low K → jagged boundary (overfits). High K → smoother (underfits).',
  k_kmeans:     'Number of clusters to find. Should match natural groupings in your data.',
  maxDepth:     'Max tree splits. Deeper → more complex boundary, higher overfitting risk.',
  nTrees:       'More trees → stabler predictions. Diminishing returns beyond ~20–30.',
  C:            'Penalty for margin violations. High C → smaller margin, fewer errors (overfits).',
  degree:       'Degree 2 = parabola, 3 = cubic. Higher → more flexible but may overfit.',
};
const PARAM_WARNINGS = {
  learningRate: v => v > 0.2 ? 'High LR may cause instability' : null,
  C:            v => v > 3   ? 'High C may overfit to noise'    : null,
  degree:       v => v > 3   ? 'High degree prone to overfitting': null,
  maxDepth:     v => v >= 7  ? 'Deep trees tend to overfit'     : null,
};

// ─── Dataset options ──────────────────────────────────────────────────────
const DATASETS = {
  regression:     [{ id: 'synth',      label: 'Synthetic Linear' },  { id: 'poly',       label: 'Polynomial Curve' }],
  classification: [{ id: 'blobs',      label: '2-Class Blobs' },     { id: 'linear_sep', label: 'Linearly Separable' },
                   { id: 'moon',       label: 'Moon Shapes' },       { id: 'spiral',     label: 'Spiral' },
                   { id: 'multiclass', label: '3-Class Blobs' },     { id: 'iris',       label: 'Iris' }],
  clustering:     [{ id: 'cluster3',   label: '3 Clusters' },        { id: 'cluster5',   label: '5 Clusters' }],
  deeplearning:   [{ id: 'spiral',     label: 'Spiral' },            { id: 'moon',       label: 'Moon Shapes' },
                   { id: 'blobs',      label: '2-Class Blobs' }],
};

function getDatasetCategory(tab) {
  if (REGRESSION_ALGOS.has(tab)) return 'regression';
  if (CLUSTERING_ALGOS.has(tab)) return 'clustering';
  if (tab === 'nn' || tab === 'dnn') return 'deeplearning';
  return 'classification';
}

function generateData(datasetId, tab) {
  switch (datasetId) {
    case 'synth':       return generateLinearData(60);
    case 'poly':        return generatePolynomialData(60);
    case 'blobs':       return generateClassificationData(70);
    case 'linear_sep':  return generateLinearlySeperableData(60);
    case 'moon':        return generateMoonData(100);
    case 'spiral':      return generateSpiralData(100);
    case 'multiclass':  return generateMultiClassData(90);
    case 'iris':        return generateIrisData();
    case 'cluster3':    return generateClusteringData(70, 3);
    case 'cluster5':    return generateClusteringData5(100);
    default:
      if (REGRESSION_ALGOS.has(tab)) return generateLinearData(60);
      if (CLUSTERING_ALGOS.has(tab)) return generateClusteringData(70, 3);
      return generateClassificationData(70);
  }
}

// ─── Train a single algorithm (for comparison mode) ───────────────────────
async function trainAlgo(tab, data, params, stopRef) {
  if (tab === 'linear') {
    return trainLinearRegression(data, params, async () => {});
  } else if (tab === 'poly') {
    return trainPolynomialRegression(data, { ...params }, async () => {});
  } else if (tab === 'ridge') {
    return trainRidgeRegression(data, params, async () => {});
  } else if (tab === 'logistic') {
    return trainLogisticRegression(data, params, async () => {});
  } else if (tab === 'svm') {
    return trainSVM(data, params, async () => {});
  } else if (tab === 'knn') {
    const surface = buildKNNSurface(data, params.k || 3, 35);
    return { ...surface, type: 'knn' };
  } else if (tab === 'naiveBayes') {
    return trainNaiveBayes(data);
  } else if (tab === 'decisionTree') {
    return trainDecisionTree(data, { maxDepth: params.maxDepth });
  } else if (tab === 'randomForest') {
    return trainRandomForest(data, { nTrees: params.nTrees, maxDepth: params.maxDepth });
  } else if (tab === 'nn' || tab === 'dnn') {
    const hiddenLayers = tab === 'dnn' ? [8, 8, 4] : [4, 4];
    return trainFFNN(data, params, { hiddenLayers, activation: 'relu' }, async () => {});
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
export default function Home() {
  const [activeTab,      setActiveTab]      = useState('linear');
  const [activeDataset,  setActiveDataset]  = useState('synth');
  const [data,           setData]           = useState([]);
  const [params,         setParams]         = useState({
    learningRate: 0.01, epochs: 100, k: 3, maxDepth: 4, nTrees: 10, C: 1.0, degree: 2,
  });
  const [modelResult,    setModelResult]    = useState(null);
  const [fullModel,      setFullModel]      = useState(null);
  const [nnWeights,      setNnWeights]      = useState(null);
  const [isTraining,     setIsTraining]     = useState(false);
  const [progress,       setProgress]       = useState(0);
  const [currentEpoch,   setCurrentEpoch]   = useState(null);
  const [liveMetrics,    setLiveMetrics]    = useState({ loss: null, acc: null });
  const [showAlgoMenu,   setShowAlgoMenu]   = useState(false);
  const [showExplainer,  setShowExplainer]  = useState(false);

  // Feature 1 – Training history for loss/acc chart
  const [trainingHistory, setTrainingHistory] = useState({ loss: [], acc: [], valLoss: [], valAcc: [] });

  // Feature 3 – Pause / Speed control
  const [trainingSpeed,  setTrainingSpeed]  = useState(0);   // extra ms per epoch
  const [isPaused,       setIsPaused]       = useState(false);
  const pauseRef   = useRef(false);
  const speedRef   = useRef(0);
  const stopRef    = useRef(false);
  const guideRef   = useRef(null);           // keyboard → guide nav

  // Responsive visualization sizing
  const vizContainerRef = useRef(null);
  const [vizDims, setVizDims] = useState({ w: 800, h: 520 });

  // Feature 10 – Colorblind-friendly palette toggle
  const [colorblind, setColorblind] = useState(false);

  // Feature 5 – Comparison mode
  const [comparisonMode,  setComparisonMode]  = useState(false);
  const [compareTab,      setCompareTab]      = useState('logistic');
  const [compareResult,   setCompareResult]   = useState(null);
  const [showCompareMenu, setShowCompareMenu] = useState(false);

  const category       = getDatasetCategory(activeTab);
  const datasetOptions = DATASETS[category] || DATASETS.classification;
  const currentAlgo    = ALL_ALGOS.find(a => a.id === activeTab) || ALL_ALGOS[0];
  const compareAlgo    = ALL_ALGOS.find(a => a.id === compareTab) || ALL_ALGOS[1];

  // Algorithms in the same category as the primary (for compare picker)
  const comparableAlgos = useMemo(() => {
    const group = Object.values(ALGORITHMS).find(items => items.some(a => a.id === activeTab)) ?? [];
    return group.filter(a => a.id !== activeTab);
  }, [activeTab]);

  // Auto-select a sensible compare target when primary changes
  useEffect(() => {
    if (comparableAlgos.length) setCompareTab(comparableAlgos[0].id);
  }, [activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync dataset category when algorithm changes
  useEffect(() => {
    const opts = DATASETS[getDatasetCategory(activeTab)] || [];
    setActiveDataset(opts[0]?.id || 'synth');
  }, [activeTab]);

  // Full reset on algo or dataset switch
  useEffect(() => { handleReset(); }, [activeTab, activeDataset]); // eslint-disable-line react-hooks/exhaustive-deps

  // Feature 10 – Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      // Ignore when focus is inside an input / select
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        if (!isTraining) handleTrain();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        guideRef.current?.prev();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        guideRef.current?.next();
      } else if (e.key === 'r' || e.key === 'R') {
        if (!isTraining) handleReset();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isTraining]); // eslint-disable-line react-hooks/exhaustive-deps

  // Measure visualization container so charts fill it exactly
  useEffect(() => {
    const el = vizContainerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setVizDims({ w: Math.floor(width), h: Math.floor(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const handleReset = () => {
    stopRef.current = true;
    setData(generateData(activeDataset, activeTab));
    setModelResult(null); setFullModel(null); setNnWeights(null);
    setProgress(0); setCurrentEpoch(null);
    setLiveMetrics({ loss: null, acc: null });
    setIsTraining(false);
    setTrainingHistory({ loss: [], acc: [], valLoss: [], valAcc: [] });
    setCompareResult(null);
    setIsPaused(false); pauseRef.current = false;
  };

  const handleExport = () => { if (fullModel) exportModel(fullModel, `${activeTab}-model`); };

  const handleShare = () => {
    const url = new URL(window.location.href);
    url.searchParams.set('algo', activeTab);
    url.searchParams.set('lr', params.learningRate);
    navigator.clipboard.writeText(url.toString());
    alert('Experiment link copied to clipboard!');
  };

  const handlePause = () => {
    setIsPaused(v => { pauseRef.current = !v; return !v; });
  };

  // ── Epoch callback (Feature 1 + 3 together) ───────────────────────────────
  const makeEpochCB = (totalEpochs) => async (epoch, logs) => {
    if (stopRef.current) return;

    const pct     = Math.round(((epoch + 1) / totalEpochs) * 100);
    const accVal  = logs.acc ?? logs.accuracy;
    const valLoss = logs.val_loss   ?? null;
    const valAcc  = logs.val_acc ?? logs.val_accuracy ?? null;

    setProgress(pct);
    setCurrentEpoch(epoch);
    const isRegression = REGRESSION_ALGOS.has(activeTab);
    setLiveMetrics({
      loss: logs.loss != null ? logs.loss.toFixed(4) : null,
      acc:  accVal    != null
              ? isRegression
                ? accVal.toFixed(3)                        // R² as 0.000
                : (accVal * 100).toFixed(1) + '%'          // accuracy as 87.3%
              : null,
    });
    setModelResult({ ...logs });

    // Accumulate history for TrainingChart (train + val)
    if (logs.loss != null) {
      setTrainingHistory(prev => ({
        loss:    [...prev.loss, logs.loss],
        acc:     accVal  != null ? [...prev.acc,    accVal]  : prev.acc,
        valLoss: valLoss != null ? [...(prev.valLoss ?? []), valLoss] : prev.valLoss,
        valAcc:  valAcc  != null ? [...(prev.valAcc  ?? []), valAcc]  : prev.valAcc,
      }));
    }

    // Pause: hold here until resumed
    while (pauseRef.current && !stopRef.current) {
      await new Promise(r => setTimeout(r, 80));
    }

    // Speed delay
    if (speedRef.current > 0) {
      await new Promise(r => setTimeout(r, speedRef.current));
    }
  };

  // ── Main Train ─────────────────────────────────────────────────────────────
  const handleTrain = async () => {
    if (isTraining) return;
    stopRef.current  = false;
    pauseRef.current = false;
    setIsPaused(false);
    setIsTraining(true);
    setProgress(0); setCurrentEpoch(null);
    setLiveMetrics({ loss: null, acc: null });
    setModelResult(null); setCompareResult(null);
    setTrainingHistory({ loss: [], acc: [], valLoss: [], valAcc: [] });

    const epochCB = makeEpochCB(params.epochs);

    try {
      // ─── Gradient-descent / TF.js algorithms ────────────────────────────
      if (activeTab === 'linear') {
        const r = await trainLinearRegression(data, params, epochCB);
        if (!stopRef.current) { setModelResult(r); setFullModel(r.model); saveProgress('linear', 100); }

      } else if (activeTab === 'poly') {
        const r = await trainPolynomialRegression(data, { ...params, degree: params.degree }, epochCB);
        if (!stopRef.current) { setModelResult(r); setFullModel(r.model); }

      } else if (activeTab === 'ridge') {
        const r = await trainRidgeRegression(data, params, epochCB);
        if (!stopRef.current) setModelResult(r);

      } else if (activeTab === 'logistic') {
        const r = await trainLogisticRegression(data, params, epochCB);
        if (!stopRef.current) { setModelResult(r); setFullModel(r.model); }

      } else if (activeTab === 'svm') {
        const r = await trainSVM(data, params, epochCB);
        if (!stopRef.current) setModelResult(r);

      } else if (activeTab === 'nn' || activeTab === 'dnn') {
        const hiddenLayers = activeTab === 'dnn' ? [8, 8, 4] : [4, 4];
        const nnEpochCB = async (epoch, logs) => {
          await epochCB(epoch, logs);
          if (logs.weights) setNnWeights(logs.weights);
        };
        const r = await trainFFNN(data, params, { hiddenLayers, activation: 'relu' }, nnEpochCB);
        if (!stopRef.current) {
          setNnWeights(r.weights); setFullModel(r.model); saveProgress('nn', 100);
        }

      // ─── Instant / non-epoch algorithms ─────────────────────────────────
      } else if (activeTab === 'knn') {
        setProgress(50);
        const surface    = buildKNNSurface(data, params.k || 3, 35);
        const knnCorrect = data.filter(d => parseInt(predictKNN(data, d, params.k || 3)) === d.label).length;
        setProgress(100); setCurrentEpoch(0);
        setLiveMetrics({ loss: null, acc: (knnCorrect / data.length * 100).toFixed(1) + '%' });
        setModelResult({ ...surface, type: 'knn' });

      } else if (activeTab === 'naiveBayes') {
        setProgress(50);
        const r = trainNaiveBayes(data);
        setProgress(100); setCurrentEpoch(0);
        setLiveMetrics({ loss: null, acc: (r.accuracy * 100).toFixed(1) + '%' });
        setModelResult(r);

      } else if (activeTab === 'decisionTree') {
        setProgress(50);
        const r = trainDecisionTree(data, { maxDepth: params.maxDepth });
        setProgress(100); setCurrentEpoch(0);
        setLiveMetrics({ loss: null, acc: (r.accuracy * 100).toFixed(1) + '%' });
        setModelResult(r);

      } else if (activeTab === 'randomForest') {
        const r = trainRandomForest(data, { nTrees: params.nTrees, maxDepth: params.maxDepth }, (treesDone) => {
          setProgress(Math.round(treesDone / params.nTrees * 100));
          setCurrentEpoch(treesDone - 1);
        });
        if (!stopRef.current) {
          setLiveMetrics({ loss: null, acc: (r.accuracy * 100).toFixed(1) + '%' });
          setModelResult(r);
        }

      } else if (activeTab === 'kmeans') {
        let stepResult = null;
        trainKMeans(data, { k: params.k || 3 }, (iter, { centroids, assignments }) => {
          setProgress(Math.round(iter / 20 * 100));
          setCurrentEpoch(iter - 1);
          const clustered = data.map((p, i) => ({ ...p, label: assignments[i] }));
          setData([...clustered]);
          stepResult = { type: 'kmeans', centroids };
          setModelResult({ type: 'kmeans', centroids });
        });
        if (!stopRef.current && stepResult) {
          setProgress(100);
        }
      }

      // ─── Feature 5: Comparison training ─────────────────────────────────
      if (comparisonMode && !stopRef.current && compareTab !== 'kmeans') {
        const cmpOut = await trainAlgo(compareTab, data, params, stopRef);
        if (!stopRef.current && cmpOut) setCompareResult(cmpOut);
      }

    } catch (e) {
      console.error('Training error:', e);
    }

    setIsTraining(false);
    setProgress(100);
  };

  // ── Feature 2: Click-to-predict ────────────────────────────────────────────
  const predictPoint = useMemo(() => {
    if (!modelResult) return null;

    // Regression
    if (REGRESSION_ALGOS.has(activeTab)) {
      if (modelResult.weight !== undefined) {
        const { weight, bias } = modelResult;
        return (x) => ({ type: 'regression', value: (weight * x + bias).toFixed(3) });
      }
      if (modelResult.type === 'poly' && modelResult.curve) {
        const curve = modelResult.curve;
        return (x) => {
          const pt = curve.reduce((best, p) => Math.abs(p.x - x) < Math.abs(best.x - x) ? p : best);
          return { type: 'regression', value: pt.y.toFixed(3) };
        };
      }
      return null;
    }

    // Clustering
    if (CLUSTERING_ALGOS.has(activeTab) && modelResult.centroids) {
      const { centroids } = modelResult;
      return (x, y) => {
        let minD = Infinity, cluster = 0;
        centroids.forEach((c, i) => {
          const d = (c.x - x) ** 2 + (c.y - y) ** 2;
          if (d < minD) { minD = d; cluster = i; }
        });
        return { type: 'cluster', label: cluster };
      };
    }

    // Classifiers with closed-form weights
    if (activeTab === 'logistic' && modelResult.weights) {
      const [w1, w2] = modelResult.weights, b = modelResult.bias;
      return (x, y) => {
        const prob = 1 / (1 + Math.exp(-(w1 * x + w2 * y + b)));
        return { type: 'class', label: prob >= 0.5 ? 1 : 0, confidence: Math.max(prob, 1 - prob) };
      };
    }
    if (activeTab === 'svm' && modelResult.weights) {
      const [w1, w2] = modelResult.weights, b = modelResult.bias;
      return (x, y) => ({
        type: 'class', label: (w1 * x + w2 * y + b) >= 0 ? 1 : 0, confidence: null,
      });
    }

    // Naive Bayes has its own embedded predict function
    if (activeTab === 'naiveBayes' && modelResult.predict) {
      const predictFn = modelResult.predict;
      return (x, y) => ({ type: 'class', label: predictFn(x, y), confidence: null });
    }

    // Surface-cell lookup (KNN, DTree, RF)
    if (modelResult.cells) {
      const cells = modelResult.cells;
      return (x, y) => {
        let minD = Infinity, label = 0, prob = null;
        cells.forEach(cell => {
          const d = (cell.x - x) ** 2 + (cell.y - y) ** 2;
          if (d < minD) {
            minD = d;
            label = cell.prob !== undefined ? (cell.prob >= 0.5 ? 1 : 0) : cell.label;
            prob  = cell.prob ?? null;
          }
        });
        return { type: 'class', label, confidence: prob != null ? Math.max(prob, 1 - prob) : null };
      };
    }

    // Neural network – synchronous TF.js prediction via dataSync()
    if ((activeTab === 'nn' || activeTab === 'dnn') && fullModel) {
      const model = fullModel;
      return (x, y) => {
        const tensor = tf.tensor2d([[x, y]]);
        const pred   = model.predict(tensor).dataSync()[0];
        tensor.dispose();
        return { type: 'class', label: pred >= 0.5 ? 1 : 0, confidence: Math.max(pred, 1 - pred) };
      };
    }

    return null;
  }, [modelResult, activeTab, fullModel]);

  // ── Feature 4: Confusion matrix (derived from predictPoint + data) ─────────
  const confusionMatrix = useMemo(() => {
    if (!predictPoint) return null;
    if (REGRESSION_ALGOS.has(activeTab) || CLUSTERING_ALGOS.has(activeTab)) return null;
    if (!data.length) return null;
    const classIds = [...new Set(data.map(d => d.label))].sort((a, b) => a - b);
    if (classIds.length < 2 || classIds.length > 6) return null;

    const n      = classIds.length;
    const idx    = Object.fromEntries(classIds.map((c, i) => [c, i]));
    const matrix = Array.from({ length: n }, () => new Array(n).fill(0));
    data.forEach(d => {
      const res = predictPoint(d.x, d.y);
      if (!res) return;
      const actual = idx[d.label], pred = idx[res.label];
      if (actual !== undefined && pred !== undefined) matrix[actual][pred]++;
    });
    return { matrix, labels: classIds.map(String) };
  }, [predictPoint, data, activeTab]);

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <main className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans">

      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside className="w-64 bg-slate-900/60 border-r border-white/5 flex flex-col shrink-0 overflow-y-auto">
        <div className="px-6 pt-6 pb-4 border-b border-white/5">
          <h1 className="text-lg font-black gradient-text tracking-tight">MLPlayground</h1>
          <p className="text-[11px] text-slate-500 mt-0.5">Interactive Learning Engine</p>
        </div>

        {/* Hyperparameters */}
        <div className="p-4 border-b border-white/5">
          <div className="flex items-center gap-2 text-xs text-slate-400 mb-2">
            <Settings2 size={13} /><span className="font-semibold">Hyperparameters</span>
          </div>

          {NEEDS_EPOCH.has(activeTab) && (<>
            <ParameterControl label="Learning Rate" value={params.learningRate} min={0.0001} max={0.5} step={0.0001}
              hint={PARAM_HINTS.learningRate} warning={PARAM_WARNINGS.learningRate}
              onChange={v => setParams(p => ({ ...p, learningRate: parseFloat(v) }))} />
            <ParameterControl label="Epochs" value={params.epochs} min={10} max={500} step={10}
              hint={PARAM_HINTS.epochs}
              onChange={v => setParams(p => ({ ...p, epochs: parseInt(v) }))} />
          </>)}
          {activeTab === 'poly' && (
            <ParameterControl label="Degree" value={params.degree} min={2} max={5} step={1}
              hint={PARAM_HINTS.degree} warning={PARAM_WARNINGS.degree}
              onChange={v => setParams(p => ({ ...p, degree: parseInt(v) }))} />
          )}
          {activeTab === 'knn' && (
            <ParameterControl label="K (Neighbors)" value={params.k} min={1} max={15} step={1}
              hint={PARAM_HINTS.k_knn}
              onChange={v => setParams(p => ({ ...p, k: parseInt(v) }))} />
          )}
          {activeTab === 'kmeans' && (
            <ParameterControl label="K (Clusters)" value={params.k} min={2} max={8} step={1}
              hint={PARAM_HINTS.k_kmeans}
              onChange={v => setParams(p => ({ ...p, k: parseInt(v) }))} />
          )}
          {(activeTab === 'decisionTree' || activeTab === 'randomForest') && (
            <ParameterControl label="Max Depth" value={params.maxDepth} min={1} max={8} step={1}
              hint={PARAM_HINTS.maxDepth} warning={PARAM_WARNINGS.maxDepth}
              onChange={v => setParams(p => ({ ...p, maxDepth: parseInt(v) }))} />
          )}
          {activeTab === 'randomForest' && (
            <ParameterControl label="Num Trees" value={params.nTrees} min={3} max={50} step={1}
              hint={PARAM_HINTS.nTrees}
              onChange={v => setParams(p => ({ ...p, nTrees: parseInt(v) }))} />
          )}
          {activeTab === 'svm' && (
            <ParameterControl label="C (Margin)" value={params.C} min={0.1} max={5} step={0.1}
              hint={PARAM_HINTS.C} warning={PARAM_WARNINGS.C}
              onChange={v => setParams(p => ({ ...p, C: parseFloat(v) }))} />
          )}
        </div>

        {/* Feature 3: Training Speed (only for gradient-descent algos) */}
        {NEEDS_EPOCH.has(activeTab) && (
          <div className="px-4 pt-3 pb-3 border-b border-white/5">
            <div className="text-[10px] text-slate-500 mb-1.5 font-semibold uppercase tracking-widest">
              Training Speed
            </div>
            <div className="flex gap-1">
              {[{ label: '⚡ Fast', ms: 0 }, { label: '▶ Normal', ms: 80 }, { label: '🐢 Slow', ms: 400 }].map(s => (
                <button key={s.ms} onClick={() => { setTrainingSpeed(s.ms); speedRef.current = s.ms; }}
                  className={`flex-1 py-1 rounded-lg text-[10px] font-medium transition-all ${
                    trainingSpeed === s.ms
                      ? 'bg-brand-500/20 text-brand-400 border border-brand-500/30'
                      : 'bg-slate-800 text-slate-500 hover:text-slate-300'}`}>
                  {s.label}
                </button>
              ))}
            </div>
            <p className="text-[9px] text-slate-600 mt-1.5">Slow mode lets you watch each epoch step.</p>
          </div>
        )}

        {/* Dataset */}
        <div className="p-4 border-b border-white/5">
          <div className="flex items-center gap-2 text-xs text-slate-400 mb-2">
            <Database size={13} /><span className="font-semibold">Dataset</span>
          </div>
          <select value={activeDataset} onChange={e => setActiveDataset(e.target.value)}
            className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-xs text-slate-300 focus:outline-none focus:border-brand-500">
            {datasetOptions.map(d => <option key={d.id} value={d.id}>{d.label}</option>)}
          </select>
        </div>

        {/* Actions */}
        <div className="p-4 space-y-2 mt-auto">
          {/* Colorblind palette toggle */}
          <button
            onClick={() => setColorblind(v => !v)}
            title="Toggle colorblind-friendly palette"
            aria-pressed={colorblind}
            className={`w-full flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${
              colorblind
                ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                : 'bg-slate-800 border-white/10 text-slate-500 hover:text-slate-300'
            }`}>
            <Eye size={13} />
            {colorblind ? 'Colorblind mode ON' : 'Colorblind-friendly mode'}
          </button>

          <div className="flex gap-2">
            <button onClick={handleTrain} disabled={isTraining}
              aria-label="Train model (Space)"
              className="flex-1 bg-brand-500 hover:bg-brand-600 disabled:opacity-50 text-white py-2 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all shadow-lg shadow-brand-500/20 text-sm">
              <Play size={15} fill="currentColor" />{isTraining ? 'Training…' : 'Train'}
            </button>

            {/* Pause (visible only during epoch-based training) */}
            {isTraining && NEEDS_EPOCH.has(activeTab) && (
              <button onClick={handlePause}
                title={isPaused ? 'Resume' : 'Pause'}
                className={`px-2.5 rounded-xl font-semibold transition-all border ${isPaused
                  ? 'bg-amber-500/20 border-amber-500/40 text-amber-400'
                  : 'bg-slate-800 border-white/10 text-slate-400 hover:text-white'}`}>
                {isPaused ? <Play size={14} /> : <Pause size={14} />}
              </button>
            )}

            <button onClick={handleReset} disabled={isTraining}
              className="p-2.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 rounded-xl transition-all">
              <RotateCcw size={16} />
            </button>
          </div>
          <div className="flex gap-2">
            <button onClick={handleExport} disabled={!fullModel}
              className="flex-1 text-xs bg-slate-800 hover:bg-slate-700 disabled:opacity-30 text-slate-300 py-1.5 rounded-lg flex items-center justify-center gap-1.5">
              <Download size={13} /> Export
            </button>
            <button onClick={handleShare}
              className="flex-1 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 rounded-lg flex items-center justify-center gap-1.5">
              <Share2 size={13} /> Share
            </button>
          </div>

          <div className="pt-3 mt-1 border-t border-white/5 text-center">
            <p className="text-[10px] text-slate-600 leading-relaxed">
              Developed by <span className="text-slate-400 font-semibold">Dr. Himanshu Rai</span>
              <br />
              <a
                href="https://www.linkedin.com/in/himanshurai14/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 hover:text-blue-400 transition-colors"
              >
                Follow on LinkedIn
              </a>
            </p>
          </div>
        </div>
      </aside>

      {/* ── Main Area ────────────────────────────────────────────────────── */}
      <section className="flex-1 flex flex-col overflow-hidden">

        {/* Top Bar */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-slate-950/60 backdrop-blur shrink-0 relative z-20 gap-4">

          {/* Primary Algorithm Picker */}
          <div className="relative">
            <button onClick={() => setShowAlgoMenu(v => !v)}
              className="flex items-center gap-3 px-4 py-2 bg-slate-900 border border-white/10 rounded-xl hover:border-brand-500/50 transition-all">
              <div className="p-1.5 bg-brand-500/10 rounded-lg text-brand-500"><Layers size={16} /></div>
              <div className="text-left">
                <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Algorithm</div>
                <div className="text-sm font-bold text-white flex items-center gap-1.5">
                  {currentAlgo.label}
                  <ChevronDown size={13} className={`transition-transform ${showAlgoMenu ? 'rotate-180' : ''}`} />
                </div>
              </div>
            </button>

            {showAlgoMenu && (
              <div className="absolute top-full left-0 mt-2 w-72 bg-slate-900 border border-white/10 rounded-2xl shadow-2xl z-50 overflow-hidden backdrop-blur-xl">
                {Object.entries(ALGORITHMS).map(([cat, items]) => (
                  <div key={cat} className="border-b border-white/5 last:border-0 p-2">
                    <div className="px-3 py-1.5 text-[10px] font-bold text-slate-500 uppercase tracking-widest">{cat}</div>
                    {items.map(algo => (
                      <button key={algo.id} onClick={() => { setActiveTab(algo.id); setShowAlgoMenu(false); }}
                        className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all text-sm ${activeTab === algo.id ? 'bg-brand-500/10 text-brand-400' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}>
                        {algo.icon}<span className="font-medium">{algo.label}</span>
                      </button>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* "Explain" button — opens the deep-dive popup */}
          <button
            onClick={() => setShowExplainer(true)}
            title="Read a detailed explanation of this algorithm"
            className="flex items-center gap-1.5 px-3 py-2 bg-slate-900 border border-white/10 rounded-xl text-slate-400 hover:border-brand-500/50 hover:text-brand-400 transition-all text-xs font-semibold shrink-0"
          >
            <BookOpen size={13} />
            <span className="hidden sm:inline">Explain</span>
          </button>

          {/* Compare picker (shown when comparison mode is active) */}
          {comparisonMode && comparableAlgos.length > 0 && (
            <div className="relative">
              <button onClick={() => setShowCompareMenu(v => !v)}
                className="flex items-center gap-2 px-3 py-2 bg-slate-900 border border-brand-500/30 rounded-xl text-brand-400 hover:border-brand-500/60 transition-all text-sm">
                <span className="text-[10px] text-slate-500 uppercase font-bold">vs</span>
                <span className="font-semibold">{compareAlgo.label}</span>
                <ChevronDown size={12} className={`transition-transform ${showCompareMenu ? 'rotate-180' : ''}`} />
              </button>
              {showCompareMenu && (
                <div className="absolute top-full left-0 mt-2 w-56 bg-slate-900 border border-white/10 rounded-xl shadow-2xl z-50 overflow-hidden p-2">
                  {comparableAlgos.map(algo => (
                    <button key={algo.id} onClick={() => { setCompareTab(algo.id); setShowCompareMenu(false); }}
                      className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all ${compareTab === algo.id ? 'bg-brand-500/10 text-brand-400' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}>
                      {algo.icon}<span>{algo.label}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Centre description */}
          <p className="text-sm text-slate-400 max-w-xs text-center hidden xl:block flex-1">{currentAlgo.desc}</p>

          {/* Right: compare toggle + progress */}
          <div className="flex items-center gap-3 shrink-0">
            <button onClick={() => setComparisonMode(v => !v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${comparisonMode
                ? 'bg-brand-500/10 border-brand-500/30 text-brand-400'
                : 'bg-slate-900 border-white/10 text-slate-400 hover:border-white/20 hover:text-slate-200'}`}>
              <Columns2 size={13} /> Compare
            </button>

            {isTraining && (
              <div className="flex items-center gap-2">
                {isPaused && <span className="text-xs text-amber-400 font-mono animate-pulse">⏸ Paused</span>}
                <div className="w-28 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-brand-500 transition-all duration-150 rounded-full"
                    style={{ width: `${progress}%` }} />
                </div>
                <span className="text-xs font-mono text-brand-400">{progress}%</span>
              </div>
            )}
            {!isTraining && progress === 100 && (
              <span className="text-xs text-green-400 font-mono">✓ Done</span>
            )}
          </div>
        </header>

        {/* Body */}
        <div className="flex-1 flex overflow-hidden">

          {/* Center: Visualization + Code */}
          <div className="flex-1 flex flex-col overflow-hidden p-4 gap-3">

            {/* Visualization Card — fills all remaining vertical space */}
            <div className="flex-1 min-h-0 bg-slate-900/40 rounded-2xl border border-white/5 overflow-hidden p-2">
              <div ref={vizContainerRef} className="w-full h-full">
                {comparisonMode ? (
                  /* Side-by-side comparison */
                  <div className="grid grid-cols-2 gap-3 h-full">
                    <div className="flex flex-col h-full">
                      <div className="text-[10px] text-slate-400 font-semibold px-2 pb-1 flex items-center gap-1.5 shrink-0">
                        <span className="w-2 h-2 rounded-full bg-brand-500 inline-block" />
                        {currentAlgo.label}
                      </div>
                      <ScatterPlot data={data} regressionLine={modelResult} isTraining={isTraining}
                        currentEpoch={currentEpoch} totalEpochs={params.epochs}
                        predictPoint={predictPoint} colorblind={colorblind}
                        width={Math.max(200, Math.floor(vizDims.w / 2) - 10)}
                        height={Math.max(200, vizDims.h - 22)} />
                    </div>
                    <div className="flex flex-col h-full">
                      <div className="text-[10px] text-slate-400 font-semibold px-2 pb-1 flex items-center gap-1.5 shrink-0">
                        <span className="w-2 h-2 rounded-full bg-purple-400 inline-block" />
                        {compareAlgo.label}
                        {!compareResult && (
                          <span className="text-slate-600 font-normal ml-1">— train to compare</span>
                        )}
                      </div>
                      <ScatterPlot data={data} regressionLine={compareResult} isTraining={false}
                        colorblind={colorblind}
                        width={Math.max(200, Math.floor(vizDims.w / 2) - 10)}
                        height={Math.max(200, vizDims.h - 22)} />
                    </div>
                  </div>
                ) : activeTab === 'nn' || activeTab === 'dnn' ? (
                  <div className="grid grid-cols-2 gap-2 h-full">
                    <NeuralNetworkGraph weights={nnWeights}
                      layers={activeTab === 'dnn' ? [2, 8, 8, 4, 1] : [2, 4, 4, 1]}
                      width={Math.max(200, Math.floor(vizDims.w / 2) - 5)}
                      height={Math.max(200, vizDims.h)} />
                    <ScatterPlot data={data} regressionLine={modelResult} isTraining={isTraining}
                      currentEpoch={currentEpoch} totalEpochs={params.epochs}
                      predictPoint={predictPoint} colorblind={colorblind}
                      width={Math.max(200, Math.ceil(vizDims.w / 2) - 5)}
                      height={Math.max(200, vizDims.h)} />
                  </div>
                ) : (
                  <ScatterPlot data={data} regressionLine={modelResult} isTraining={isTraining}
                    currentEpoch={currentEpoch} totalEpochs={params.epochs}
                    predictPoint={predictPoint} colorblind={colorblind}
                    width={Math.max(400, vizDims.w)}
                    height={Math.max(300, vizDims.h)} />
                )}
              </div>
            </div>

            {/* Code Snippet — scrollable within fixed height */}
            <div className="shrink-0">
              <CodeSnippet activeTab={activeTab} params={params} />
            </div>
          </div>

          {/* Right Panel — wider to comfortably fit all content */}
          <aside className="w-96 shrink-0 border-l border-white/5 bg-slate-900/30 overflow-y-auto p-4 space-y-4">

            {/* Live Metrics */}
            <div className="bg-slate-900/80 rounded-xl border border-white/5 p-4">
              <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-3">Live Metrics</h3>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard label="Loss"     value={liveMetrics.loss ?? '—'} highlight={!!liveMetrics.loss} />
                <MetricCard
                  label={REGRESSION_ALGOS.has(activeTab) ? 'R² Score' : 'Accuracy'}
                  value={liveMetrics.acc ?? '—'}
                  highlight={!!liveMetrics.acc}
                />
                <MetricCard label="Progress" value={`${progress}%`}          highlight={isTraining} />
                <MetricCard label="Epoch"    value={currentEpoch !== null ? currentEpoch + 1 : '—'} highlight={isTraining} />
              </div>
              {NEEDS_EPOCH.has(activeTab) && (
                <div className="mt-3">
                  <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                    <span>Epoch Progress</span>
                    <span>{currentEpoch !== null ? currentEpoch + 1 : 0} / {params.epochs}</span>
                  </div>
                  <div className="w-full h-1 bg-slate-800 rounded-full">
                    <div className="h-full bg-brand-500 rounded-full transition-all duration-100"
                      style={{ width: `${currentEpoch !== null ? ((currentEpoch + 1) / params.epochs) * 100 : 0}%` }} />
                  </div>
                </div>
              )}
            </div>

            {/* Training History Chart — lives in right panel so viz gets full height */}
            {NEEDS_EPOCH.has(activeTab) && (
              <TrainingChart
                key={activeTab}
                history={trainingHistory}
                isRegression={REGRESSION_ALGOS.has(activeTab)}
              />
            )}

            {/* Confusion Matrix */}
            {confusionMatrix && (
              <ConfusionMatrix matrix={confusionMatrix.matrix} labels={confusionMatrix.labels} />
            )}

            {/* Feature Importance (Decision Tree & Random Forest) */}
            {modelResult?.featureImportance && (activeTab === 'decisionTree' || activeTab === 'randomForest') && (
              <FeatureImportance importance={modelResult.featureImportance} modelType={activeTab} />
            )}

            <Guide activeTab={activeTab} stepRef={guideRef} />
            <Quiz  activeTab={activeTab} />

            {/* How it works + keyboard hint */}
            <div className="bg-slate-900/80 p-4 rounded-xl border border-white/5">
              <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-2">How it works</h3>
              <p className="text-xs text-slate-400 leading-relaxed">{currentAlgo.desc}</p>
              <div className="mt-3 pt-3 border-t border-white/5 text-[10px] text-slate-500 space-y-1">
                {NEEDS_EPOCH.has(activeTab)   && <div>📈 Watch the curve animate each epoch</div>}
                {NEEDS_EPOCH.has(activeTab)   && <div>⏸ Pause to inspect any epoch in detail</div>}
                {predictPoint                  && <div>🎯 Click the plot to predict any point</div>}
                {confusionMatrix               && <div>🟩 Confusion matrix shows per-class accuracy</div>}
                {(activeTab === 'knn' || activeTab === 'naiveBayes' || activeTab === 'decisionTree' || activeTab === 'randomForest') && <div>🗺 Decision regions shown in background</div>}
                {activeTab === 'kmeans'        && <div>⭐ Stars show centroids moving each step</div>}
                {(activeTab === 'nn' || activeTab === 'dnn') && <div>🔥 Neural net heatmap + graph live</div>}
                <div className="border-t border-white/5 pt-1 mt-1">
                  <span className="text-slate-600">⌨ </span>Space = train · ←/→ = guide · R = reset
                </div>
              </div>
            </div>
          </aside>
        </div>
      </section>

      {/* Algorithm deep-dive explainer popup */}
      {showExplainer && (
        <AlgoExplainer algoId={activeTab} onClose={() => setShowExplainer(false)} />
      )}
    </main>
  );
}

function MetricCard({ label, value, highlight }) {
  return (
    <div className={`p-3 rounded-lg border transition-all ${highlight ? 'bg-brand-500/5 border-brand-500/20' : 'bg-slate-950/50 border-white/5'}`}>
      <div className="text-[10px] text-slate-500 mb-0.5">{label}</div>
      <div className={`text-sm font-mono font-bold ${highlight ? 'text-brand-400' : 'text-slate-300'}`}>{value}</div>
    </div>
  );
}
