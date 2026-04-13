"use client";
import React, { useEffect, useRef } from 'react';

// Registers Chart.js only once across the module
let chartJsReady = false;
let Chart;

async function getChart() {
  if (!chartJsReady) {
    const mod = await import('chart.js/auto');
    Chart = mod.Chart ?? mod.default;
    chartJsReady = true;
  }
  return Chart;
}

const TrainingChart = ({ history, isRegression = false }) => {
  const canvasRef = useRef(null);
  const chartRef  = useRef(null);

  const hasData    = history?.loss?.length > 0;
  const hasAcc     = history?.acc?.length  > 0;
  const hasValLoss = history?.valLoss?.length > 0;
  const hasValAcc  = history?.valAcc?.length  > 0;

  // Detect overfitting: val_loss rising while train_loss is falling
  const isOverfitting = (() => {
    if (!hasValLoss || history.valLoss.length < 6) return false;
    const n    = history.valLoss.length;
    const tail = history.valLoss.slice(-Math.min(5, n));
    const head = history.loss.slice(-Math.min(5, n));
    const valTrend   = tail[tail.length - 1] - tail[0];
    const trainTrend = head[head.length - 1] - head[0];
    return valTrend > 0.005 && trainTrend < 0;
  })();

  // ── Create chart once on mount ────────────────────────────────────────────
  useEffect(() => {
    let instance;
    getChart().then(ChartCtor => {
      if (!canvasRef.current) return;
      // Destroy any leftover instance on this canvas (React StrictMode fires effects twice)
      const existing = ChartCtor.getChart(canvasRef.current);
      if (existing) existing.destroy();
      instance = new ChartCtor(canvasRef.current, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Train Loss',
              data: [],
              borderColor: '#ef4444',
              backgroundColor: 'rgba(239,68,68,0.07)',
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.35,
              fill: true,
            },
            {
              label: 'Val Loss',
              data: [],
              borderColor: '#fb923c',
              backgroundColor: 'rgba(251,146,60,0.04)',
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.35,
              borderDash: [5, 3],
              fill: false,
              hidden: true,
            },
            {
              label: isRegression ? 'Train R²' : 'Train Acc',
              data: [],
              borderColor: '#10b981',
              backgroundColor: 'rgba(16,185,129,0.07)',
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.35,
              fill: true,
              yAxisID: 'y1',
              hidden: true,
            },
            {
              label: isRegression ? 'Val R²' : 'Val Acc',
              data: [],
              borderColor: '#34d399',
              backgroundColor: 'rgba(52,211,153,0.04)',
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.35,
              borderDash: [5, 3],
              yAxisID: 'y1',
              hidden: true,
            },
          ],
        },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#1e293b',
              titleColor: '#94a3b8',
              bodyColor: '#e2e8f0',
              borderColor: '#334155',
              borderWidth: 1,
              padding: 6,
              callbacks: {
                label: ctx => {
                  const v = ctx.parsed.y;
                  return ` ${ctx.dataset.label}: ${v != null ? v.toFixed(4) : '—'}`;
                },
              },
            },
          },
          scales: {
            x: {
              title: { display: true, text: 'Epoch', color: '#475569', font: { size: 9 } },
              ticks: { color: '#475569', font: { size: 9 }, maxTicksLimit: 6, autoSkip: true },
              grid: { color: '#1e293b' },
              border: { color: '#334155' },
            },
            y: {
              title: { display: true, text: 'Loss', color: '#f87171', font: { size: 9 } },
              ticks: { color: '#475569', font: { size: 9 }, maxTicksLimit: 5 },
              grid: { color: '#1e293b' },
              border: { color: '#334155' },
            },
            y1: {
              type: 'linear',
              position: 'right',
              title: { display: true, text: isRegression ? 'R²' : 'Acc', color: '#34d399', font: { size: 9 } },
              ticks: { color: '#34d399', font: { size: 9 }, maxTicksLimit: 5 },
              grid: { display: false },
              border: { color: '#334155' },
              min: 0,
              max: 1,
              display: false,
            },
          },
        },
      });
      chartRef.current = instance;
    });

    return () => {
      instance?.destroy();
      chartRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Push data updates without recreating the chart ────────────────────────
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !history) return;

    chart.data.labels = history.loss.map((_, i) => i + 1);

    // Train loss (always shown)
    chart.data.datasets[0].data = history.loss;

    // Val loss (show when available)
    chart.data.datasets[1].data   = history.valLoss ?? [];
    chart.data.datasets[1].hidden = !hasValLoss;

    // Train accuracy
    chart.data.datasets[2].data   = history.acc ?? [];
    chart.data.datasets[2].hidden = !hasAcc;

    // Val accuracy
    chart.data.datasets[3].data   = history.valAcc ?? [];
    chart.data.datasets[3].hidden = !hasValAcc;

    // Show right y-axis only when accuracy data exists
    chart.options.scales.y1.display = hasAcc || hasValAcc;

    chart.update('none');
  }, [history, hasAcc, hasValLoss, hasValAcc]);

  return (
    <div className="bg-slate-900/80 rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">
            Training History
          </h3>
          {isOverfitting && (
            <span className="text-[10px] bg-amber-500/20 text-amber-400 border border-amber-500/30 px-2 py-0.5 rounded-full font-semibold animate-pulse">
              ⚠ Overfitting
            </span>
          )}
        </div>
        {hasData && (
          <span className="text-[10px] text-slate-500 font-mono">
            {history.loss.length} epoch{history.loss.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Canvas is always mounted so the Chart.js instance is created on mount.
          The placeholder overlays it when no data exists yet. */}
      <div className="relative">
        <div style={{ height: 150 }}>
          <canvas ref={canvasRef} />
        </div>
        {!hasData && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-600 text-xs text-center px-4 leading-relaxed bg-slate-900/80 rounded-lg">
            Train a model to see the learning curve
          </div>
        )}
      </div>

      {hasData && (
        <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-slate-500">
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-red-400 inline-block rounded" /> Train Loss
          </span>
          {hasValLoss && (
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-orange-400 inline-block rounded border-dashed border-t border-orange-400" />
              Val Loss
            </span>
          )}
          {hasAcc && (
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-emerald-400 inline-block rounded" />
              {isRegression ? 'Train R²' : 'Train Acc'}
            </span>
          )}
          {hasValAcc && (
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-teal-300 inline-block rounded" />
              {isRegression ? 'Val R²' : 'Val Acc'}
            </span>
          )}
          {hasValLoss && (
            <span className="text-slate-600 italic">
              (dashed = validation — watch for divergence)
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default TrainingChart;
