"use client";
import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Trophy, BookOpen, Target, Star, Activity, Brain } from 'lucide-react';

const ALGO_META = {
  linear:       { label: 'Linear Regression',      group: 'Regression',      color: '#38bdf8' },
  poly:         { label: 'Polynomial Regression',   group: 'Regression',      color: '#a78bfa' },
  ridge:        { label: 'Ridge Regression',        group: 'Regression',      color: '#34d399' },
  logistic:     { label: 'Logistic Regression',     group: 'Classification',  color: '#f472b6' },
  knn:          { label: 'K-Nearest Neighbors',     group: 'Classification',  color: '#60a5fa' },
  naiveBayes:   { label: 'Naive Bayes',             group: 'Classification',  color: '#10b981' },
  svm:          { label: 'Support Vector Machine',  group: 'Classification',  color: '#f59e0b' },
  decisionTree: { label: 'Decision Tree',           group: 'Classification',  color: '#fb923c' },
  randomForest: { label: 'Random Forest',           group: 'Classification',  color: '#f97316' },
  kmeans:       { label: 'K-Means Clustering',      group: 'Clustering',      color: '#e879f9' },
  nn:           { label: 'Neural Network (MLP)',    group: 'Deep Learning',   color: '#818cf8' },
  dnn:          { label: 'Deep Neural Net',         group: 'Deep Learning',   color: '#6366f1' },
  pca:          { label: 'PCA',                     group: 'Dim. Reduction',  color: '#34d399' },
};

const GROUP_ORDER = ['Regression', 'Classification', 'Clustering', 'Deep Learning', 'Dim. Reduction'];

function getRank(score) {
  if (score >= 90) return { label: 'Expert',     color: '#f59e0b', icon: '🏆' };
  if (score >= 70) return { label: 'Proficient', color: '#22c55e', icon: '🎯' };
  if (score >= 40) return { label: 'Learning',   color: '#60a5fa', icon: '📖' };
  return               { label: 'Beginner',   color: '#94a3b8', icon: '🌱' };
}

function CircleProgress({ value, color, size = 56 }) {
  const r     = (size - 6) / 2;
  const circ  = 2 * Math.PI * r;
  const dash  = (value / 100) * circ;

  return (
    <svg width={size} height={size} className="rotate-[-90deg]">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#1e293b" strokeWidth={5} />
      <circle
        cx={size / 2} cy={size / 2} r={r}
        fill="none"
        stroke={color}
        strokeWidth={5}
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
        style={{ transition: 'stroke-dasharray 0.6s ease' }}
      />
    </svg>
  );
}

export default function Dashboard() {
  const [progress,   setProgress]   = useState({});
  const [flashcards, setFlashcards] = useState({});
  const [mounted,    setMounted]    = useState(false);

  useEffect(() => {
    setProgress(  JSON.parse(localStorage.getItem('ml_progress')   || '{}'));
    setFlashcards(JSON.parse(localStorage.getItem('ml_flashcards') || '{}'));
    setMounted(true);
  }, []);

  if (!mounted) return null;

  const algoIds      = Object.keys(ALGO_META);
  const quizScores   = algoIds.map(id => progress[id] || 0);
  const avgScore     = quizScores.reduce((s, v) => s + v, 0) / algoIds.length;
  const explored     = algoIds.filter(id => progress[id] > 0).length;
  const perfectCount = algoIds.filter(id => progress[id] >= 100).length;

  // Flashcard stats
  const fcEntries     = Object.entries(flashcards);
  const fcDue         = fcEntries.filter(([, d]) => new Date(d.nextReview) <= new Date()).length;

  const grouped = GROUP_ORDER.map(group => ({
    group,
    algos: algoIds.filter(id => ALGO_META[id].group === group),
  }));

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 font-sans">
      {/* Header */}
      <header className="border-b border-white/5 px-6 py-4 flex items-center gap-4 bg-slate-900/60 sticky top-0 z-10 backdrop-blur">
        <Link href="/" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
          <ArrowLeft size={16} />
          <span className="text-sm font-medium">Back to Playground</span>
        </Link>
        <div className="w-px h-5 bg-white/10" />
        <h1 className="text-base font-black gradient-text">Learning Dashboard</h1>
      </header>

      <div className="max-w-5xl mx-auto p-6 space-y-6">

        {/* Summary Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { icon: <Activity size={18} />,  label: 'Avg Quiz Score',    value: `${avgScore.toFixed(0)}%`,    color: '#38bdf8' },
            { icon: <Brain    size={18} />,  label: 'Algorithms Tried',  value: `${explored} / ${algoIds.length}`, color: '#a78bfa' },
            { icon: <Trophy   size={18} />,  label: 'Perfect Scores',    value: perfectCount,                  color: '#f59e0b' },
            { icon: <BookOpen size={18} />,  label: 'Flashcards Due',    value: fcDue,                         color: '#f472b6' },
          ].map(({ icon, label, value, color }) => (
            <div key={label} className="bg-slate-900/80 rounded-xl border border-white/5 p-4 flex items-center gap-3">
              <div className="p-2 rounded-lg" style={{ background: color + '18', color }}>
                {icon}
              </div>
              <div>
                <div className="text-[10px] text-slate-500 mb-0.5">{label}</div>
                <div className="text-lg font-bold font-mono" style={{ color }}>{value}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Overall Progress Bar */}
        <div className="bg-slate-900/80 rounded-xl border border-white/5 p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-bold text-slate-300">Overall Progress</h2>
            <span className="text-xs font-mono text-slate-400">{avgScore.toFixed(1)}% average</span>
          </div>
          <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{ width: `${avgScore}%`, background: 'linear-gradient(90deg, #6366f1, #a78bfa)' }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-600 mt-1">
            <span>Beginner</span><span>Proficient</span><span>Expert</span>
          </div>
        </div>

        {/* Per-Algorithm Breakdown */}
        {grouped.map(({ group, algos }) => (
          <div key={group} className="bg-slate-900/80 rounded-xl border border-white/5 p-5">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">{group}</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {algos.map(id => {
                const meta  = ALGO_META[id];
                const score = progress[id] || 0;
                const rank  = getRank(score);
                return (
                  <div
                    key={id}
                    className="flex items-center gap-4 p-3 rounded-lg border transition-all hover:border-white/10"
                    style={{ borderColor: score > 0 ? meta.color + '30' : '#1e293b', background: score > 0 ? meta.color + '06' : 'transparent' }}
                  >
                    {/* Circle progress */}
                    <div className="relative shrink-0">
                      <CircleProgress value={score} color={meta.color} />
                      <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold font-mono" style={{ color: meta.color }}>
                        {score}%
                      </span>
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-semibold text-slate-200 truncate">{meta.label}</div>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-[10px]" style={{ color: rank.color }}>
                          {rank.icon} {rank.label}
                        </span>
                        {score === 0 && (
                          <Link href="/" className="text-[10px] text-slate-600 hover:text-brand-400 transition-colors underline">
                            Start learning →
                          </Link>
                        )}
                      </div>
                    </div>

                    {/* Score bar */}
                    <div className="shrink-0 w-16">
                      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-700"
                          style={{ width: `${score}%`, background: meta.color }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        {/* Flashcard Stats */}
        {fcEntries.length > 0 && (
          <div className="bg-slate-900/80 rounded-xl border border-white/5 p-5">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
              <Star size={13} className="text-amber-400" />
              Flashcard Spaced Repetition
            </h2>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold font-mono text-amber-400">{fcEntries.length}</div>
                <div className="text-[10px] text-slate-500 mt-1">Total cards seen</div>
              </div>
              <div>
                <div className="text-2xl font-bold font-mono text-red-400">{fcDue}</div>
                <div className="text-[10px] text-slate-500 mt-1">Due for review</div>
              </div>
              <div>
                <div className="text-2xl font-bold font-mono text-green-400">{fcEntries.length - fcDue}</div>
                <div className="text-[10px] text-slate-500 mt-1">Up to date</div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center pb-4">
          <p className="text-[11px] text-slate-600">
            Progress is stored in your browser. Keep learning to level up!
          </p>
          <Link href="/" className="mt-2 inline-block text-xs text-brand-400 hover:text-brand-300 transition-colors">
            ← Back to ML Playground
          </Link>
        </div>
      </div>
    </main>
  );
}
