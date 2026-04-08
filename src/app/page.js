"use client";
import React, { useState, useEffect, useRef } from 'react';
import {
  Play, RotateCcw, Box, Activity, Settings2, BookOpen,
  Download, TerminalSquare, Share2, Database, ChevronDown,
  Layers, GitBranch, Cpu, Zap, Target, Grid3x3
} from 'lucide-react';
import CodeSnippet from '@/components/CodeSnippet';
import NeuralNetworkGraph from '@/components/visualizers/NeuralNetworkGraph';
import ScatterPlot from '@/components/visualizers/ScatterPlot';
import ParameterControl from '@/components/ParameterControl';
import Guide from '@/components/Guide';
import Quiz from '@/components/Quiz';
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
    { id: 'linear',     label: 'Linear Regression',     icon: <Activity size={15} />,    desc: 'Fit a straight line to predict continuous values.' },
    { id: 'poly',       label: 'Polynomial Regression', icon: <Zap size={15} />,         desc: 'Fit a curved polynomial to capture non-linear trends.' },
    { id: 'ridge',      label: 'Ridge Regression',      icon: <Target size={15} />,      desc: 'Linear regression with L2 regularization to prevent overfitting.' },
  ],
  "Classification": [
    { id: 'logistic',   label: 'Logistic Regression',   icon: <Activity size={15} />,    desc: 'Binary classification using a sigmoid decision boundary.' },
    { id: 'knn',        label: 'K-Nearest Neighbors',   icon: <BookOpen size={15} />,    desc: 'Classify points by majority vote of K nearest neighbors.' },
    { id: 'naiveBayes', label: 'Naive Bayes',           icon: <Grid3x3 size={15} />,     desc: 'Probabilistic classifier based on Bayes\' theorem.' },
    { id: 'svm',        label: 'Support Vector Machine',icon: <Target size={15} />,      desc: 'Find the maximum-margin hyperplane that separates classes.' },
    { id: 'decisionTree',label: 'Decision Tree',        icon: <GitBranch size={15} />,   desc: 'Recursively partition data using feature thresholds.' },
    { id: 'randomForest',label: 'Random Forest',        icon: <Layers size={15} />,      desc: 'Ensemble of decision trees that vote on the final class.' },
  ],
  "Clustering": [
    { id: 'kmeans',     label: 'K-Means Clustering',    icon: <Activity size={15} />,    desc: 'Partition unlabeled data into K groups by centroid proximity.' },
  ],
  "Deep Learning": [
    { id: 'nn',         label: 'Neural Network (MLP)',  icon: <Box size={15} />,         desc: 'Multi-layer perceptron that learns non-linear decision boundaries.' },
    { id: 'dnn',        label: 'Deep Neural Net',       icon: <Cpu size={15} />,         desc: 'Deep MLP with more hidden layers for complex patterns.' },
  ]
};

const ALL_ALGOS = Object.values(ALGORITHMS).flat();

const REGRESSION_ALGOS = new Set(['linear', 'poly', 'ridge']);
const CLUSTERING_ALGOS = new Set(['kmeans']);
const NEEDS_EPOCH = new Set(['linear', 'poly', 'ridge', 'logistic', 'svm', 'nn', 'dnn']);

// Dataset options per category
const DATASETS = {
  regression:     [{ id: 'synth',    label: 'Synthetic Linear' }, { id: 'poly',     label: 'Polynomial Curve' }],
  classification: [{ id: 'blobs',    label: '2-Class Blobs' }, { id: 'linear_sep',  label: 'Linearly Separable' }, { id: 'moon',     label: 'Moon Shapes' }, { id: 'spiral',   label: 'Spiral' }, { id: 'multiclass', label: '3-Class Blobs' }, { id: 'iris',     label: 'Iris' }],
  clustering:     [{ id: 'cluster3', label: '3 Clusters' }, { id: 'cluster5',  label: '5 Clusters' }],
  deeplearning:   [{ id: 'spiral',   label: 'Spiral' }, { id: 'moon',     label: 'Moon Shapes' }, { id: 'blobs',    label: '2-Class Blobs' }],
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

// ─────────────────────────────────────────────────────────────────────────────
export default function Home() {
  const [activeTab, setActiveTab] = useState('linear');
  const [activeDataset, setActiveDataset] = useState('synth');
  const [data, setData] = useState([]);
  const [params, setParams] = useState({ learningRate: 0.01, epochs: 100, k: 3, maxDepth: 4, nTrees: 10, C: 1.0, degree: 2 });
  const [modelResult, setModelResult] = useState(null);
  const [fullModel, setFullModel] = useState(null);
  const [nnWeights, setNnWeights] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(null);
  const [liveMetrics, setLiveMetrics] = useState({ loss: null, acc: null });
  const [showAlgoMenu, setShowAlgoMenu] = useState(false);
  const stopRef = useRef(false);

  const category = getDatasetCategory(activeTab);
  const datasetOptions = DATASETS[category] || DATASETS.classification;

  // Reset dataset to a sensible default when algo changes
  useEffect(() => {
    const opts = DATASETS[getDatasetCategory(activeTab)] || [];
    setActiveDataset(opts[0]?.id || 'synth');
  }, [activeTab]);

  useEffect(() => {
    handleReset();
  }, [activeTab, activeDataset]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleReset = () => {
    stopRef.current = true;
    setData(generateData(activeDataset, activeTab));
    setModelResult(null);
    setFullModel(null);
    setNnWeights(null);
    setProgress(0);
    setCurrentEpoch(null);
    setLiveMetrics({ loss: null, acc: null });
    setIsTraining(false);
  };

  const handleExport = () => { if (fullModel) exportModel(fullModel, `${activeTab}-model`); };

  const handleShare = () => {
    const url = new URL(window.location.href);
    url.searchParams.set('algo', activeTab);
    url.searchParams.set('lr', params.learningRate);
    navigator.clipboard.writeText(url.toString());
    alert('Experiment link copied to clipboard!');
  };

  const handleTrain = async () => {
    if (isTraining) return;
    stopRef.current = false;
    setIsTraining(true);
    setProgress(0);
    setCurrentEpoch(null);
    setLiveMetrics({ loss: null, acc: null });
    setModelResult(null);

    const epochCB = (epoch, logs) => {
      if (stopRef.current) return;
      const pct = Math.round(((epoch + 1) / params.epochs) * 100);
      setProgress(pct);
      setCurrentEpoch(epoch);
      setLiveMetrics({ loss: logs.loss?.toFixed(4) ?? null, acc: logs.acc ? (logs.acc * 100).toFixed(1) + '%' : null });
      setModelResult({ ...logs });
    };

    try {
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

      } else if (activeTab === 'knn') {
        setProgress(50);
        const surface = buildKNNSurface(data, params.k || 3, 35);
        setProgress(100); setCurrentEpoch(0);
        setModelResult({ ...surface, type: 'knn' });

      } else if (activeTab === 'naiveBayes') {
        setProgress(50);
        const r = trainNaiveBayes(data);
        setProgress(100); setCurrentEpoch(0);
        setModelResult(r);

      } else if (activeTab === 'decisionTree') {
        setProgress(50);
        const r = trainDecisionTree(data, { maxDepth: params.maxDepth });
        setProgress(100); setCurrentEpoch(0);
        setModelResult(r);

      } else if (activeTab === 'randomForest') {
        const r = trainRandomForest(data, { nTrees: params.nTrees, maxDepth: params.maxDepth }, (treesDone) => {
          setProgress(Math.round(treesDone / params.nTrees * 100));
          setCurrentEpoch(treesDone - 1);
        });
        if (!stopRef.current) setModelResult(r);

      } else if (activeTab === 'kmeans') {
        let stepResult = null;
        const r = trainKMeans(data, { k: params.k || 3 }, (iter, { centroids, assignments }) => {
          setProgress(Math.round(iter / 20 * 100));
          setCurrentEpoch(iter - 1);
          const clusteredData = data.map((p, i) => ({ ...p, label: assignments[i] }));
          setData([...clusteredData]);
          stepResult = { type: 'kmeans', centroids };
          setModelResult({ type: 'kmeans', centroids });
        });
        if (!stopRef.current) { setData([...r.clusteredData]); setModelResult({ type: 'kmeans', centroids: r.centroids }); setProgress(100); }

      } else if (activeTab === 'nn' || activeTab === 'dnn') {
        const hiddenLayers = activeTab === 'dnn' ? [8, 8, 4] : [4, 4];
        const nnEpochCB = (epoch, logs) => {
          epochCB(epoch, logs);
          if (logs.weights) setNnWeights(logs.weights);
        };
        const r = await trainFFNN(data, params, { hiddenLayers, activation: 'relu' }, nnEpochCB);
        if (!stopRef.current) {
          setNnWeights(r.weights);
          setFullModel(r.model);
          saveProgress('nn', 100);
        }
      }
    } catch (e) {
      console.error('Training error:', e);
    }

    setIsTraining(false);
    setProgress(100);
  };

  const currentAlgo = ALL_ALGOS.find(a => a.id === activeTab) || ALL_ALGOS[0];

  return (
    <main className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans">
      {/* ── Sidebar ─────────────────────────────────── */}
      <aside className="w-72 bg-slate-900/60 border-r border-white/5 flex flex-col shrink-0 overflow-y-auto">
        <div className="px-6 pt-6 pb-4 border-b border-white/5">
          <h1 className="text-lg font-black gradient-text tracking-tight">IntelliLearn ML</h1>
          <p className="text-[11px] text-slate-500 mt-0.5">Interactive Learning Engine</p>
        </div>

        {/* Params */}
        <div className="p-4 border-b border-white/5 space-y-3">
          <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
            <Settings2 size={13} /><span className="font-semibold">Hyperparameters</span>
          </div>

          {NEEDS_EPOCH.has(activeTab) && (
            <>
              <ParameterControl label="Learning Rate" value={params.learningRate} min={0.0001} max={0.5} step={0.0001}
                onChange={v => setParams(p => ({ ...p, learningRate: parseFloat(v) }))} />
              <ParameterControl label="Epochs" value={params.epochs} min={10} max={500} step={10}
                onChange={v => setParams(p => ({ ...p, epochs: parseInt(v) }))} />
            </>
          )}
          {(activeTab === 'poly') && (
            <ParameterControl label="Degree" value={params.degree} min={2} max={5} step={1}
              onChange={v => setParams(p => ({ ...p, degree: parseInt(v) }))} />
          )}
          {(activeTab === 'knn') && (
            <ParameterControl label="K (Neighbors)" value={params.k} min={1} max={15} step={1}
              onChange={v => setParams(p => ({ ...p, k: parseInt(v) }))} />
          )}
          {(activeTab === 'kmeans') && (
            <ParameterControl label="K (Clusters)" value={params.k} min={2} max={8} step={1}
              onChange={v => setParams(p => ({ ...p, k: parseInt(v) }))} />
          )}
          {(activeTab === 'decisionTree' || activeTab === 'randomForest') && (
            <ParameterControl label="Max Depth" value={params.maxDepth} min={1} max={8} step={1}
              onChange={v => setParams(p => ({ ...p, maxDepth: parseInt(v) }))} />
          )}
          {activeTab === 'randomForest' && (
            <ParameterControl label="Num Trees" value={params.nTrees} min={3} max={50} step={1}
              onChange={v => setParams(p => ({ ...p, nTrees: parseInt(v) }))} />
          )}
          {activeTab === 'svm' && (
            <ParameterControl label="C (Margin)" value={params.C} min={0.1} max={5} step={0.1}
              onChange={v => setParams(p => ({ ...p, C: parseFloat(v) }))} />
          )}
        </div>

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
          <div className="flex gap-2">
            <button onClick={handleTrain} disabled={isTraining}
              className="flex-1 bg-brand-500 hover:bg-brand-600 disabled:opacity-50 text-white py-2 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all shadow-lg shadow-brand-500/20 text-sm">
              <Play size={15} fill="currentColor" />{isTraining ? 'Training…' : 'Train'}
            </button>
            <button onClick={handleReset}
              className="p-2.5 bg-slate-800 hover:bg-slate-700 rounded-xl transition-all">
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
        </div>
      </aside>

      {/* ── Main Area ───────────────────────────────── */}
      <section className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <header className="flex items-center justify-between px-8 py-4 border-b border-white/5 bg-slate-950/60 backdrop-blur shrink-0">
          {/* Algorithm Picker */}
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

          {/* Algorithm describe */}
          <p className="text-sm text-slate-400 max-w-sm text-center hidden xl:block">{currentAlgo.desc}</p>

          {/* Progress */}
          <div className="flex items-center gap-4 min-w-[180px] justify-end">
            {isTraining && (
              <div className="flex items-center gap-3">
                <div className="w-32 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-brand-500 transition-all duration-150 rounded-full" style={{ width: `${progress}%` }} />
                </div>
                <span className="text-xs font-mono text-brand-400">{progress}%</span>
              </div>
            )}
            {!isTraining && progress === 100 && <span className="text-xs text-green-400 font-mono">✓ Done</span>}
          </div>
        </header>

        {/* Body */}
        <div className="flex-1 flex gap-0 overflow-hidden">
          {/* Visualization + Code */}
          <div className="flex-1 flex flex-col overflow-y-auto p-6 gap-6">
            {/* Chart Card */}
            <div className="bg-slate-900/40 rounded-2xl border border-white/5 overflow-hidden p-2">
              {activeTab === 'nn' || activeTab === 'dnn' ? (
                <div className="grid grid-cols-2 gap-2">
                  <NeuralNetworkGraph weights={nnWeights} layers={activeTab === 'dnn' ? [2, 8, 8, 4, 1] : [2, 4, 4, 1]} width={380} height={420} />
                  <ScatterPlot data={data} regressionLine={modelResult} isTraining={isTraining}
                    currentEpoch={currentEpoch} totalEpochs={params.epochs} width={420} height={420} />
                </div>
              ) : (
                <ScatterPlot data={data} regressionLine={modelResult} isTraining={isTraining}
                  currentEpoch={currentEpoch} totalEpochs={params.epochs} width={800} height={480} />
              )}
            </div>

            <CodeSnippet activeTab={activeTab} params={params} />
          </div>

          {/* Right Panel */}
          <aside className="w-72 shrink-0 border-l border-white/5 bg-slate-900/30 overflow-y-auto p-5 space-y-5">
            {/* Live Metrics */}
            <div className="bg-slate-900/80 rounded-xl border border-white/5 p-4">
              <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-3">Live Metrics</h3>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard label="Loss" value={liveMetrics.loss ?? '—'} highlight={!!liveMetrics.loss} />
                <MetricCard label="Accuracy" value={liveMetrics.acc ?? '—'} highlight={!!liveMetrics.acc} />
                <MetricCard label="Progress" value={`${progress}%`} highlight={isTraining} />
                <MetricCard label="Epoch" value={currentEpoch !== null ? currentEpoch + 1 : '—'} highlight={isTraining} />
              </div>

              {/* Mini epoch progress bar */}
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

            <Guide activeTab={activeTab} />
            <Quiz activeTab={activeTab} />

            {/* Algorithm Spotlight */}
            <div className="bg-slate-900/80 p-4 rounded-xl border border-white/5">
              <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-2">How it works</h3>
              <p className="text-xs text-slate-400 leading-relaxed">{currentAlgo.desc}</p>
              <div className="mt-3 pt-3 border-t border-white/5 text-[10px] text-slate-500 space-y-1">
                {NEEDS_EPOCH.has(activeTab) && <div>📈 Watch the line animate after each epoch</div>}
                {(activeTab === 'knn' || activeTab === 'naiveBayes' || activeTab === 'decisionTree' || activeTab === 'randomForest') && <div>🗺 Decision regions visualized in background</div>}
                {activeTab === 'kmeans' && <div>⭐ Stars show cluster centroids moving each step</div>}
                {(activeTab === 'nn' || activeTab === 'dnn') && <div>🔥 Neural net heatmap + graph updated live</div>}
              </div>
            </div>
          </aside>
        </div>
      </section>
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
