"use client";
import React, { useState, useEffect } from 'react';
import { HelpCircle, CheckCircle2, XCircle, Trophy } from 'lucide-react';
import { ALGORITHM_QUIZZES } from '@/lib/learning-content';
import { saveProgress } from '@/lib/persistence';

const Quiz = ({ activeTab }) => {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [selected, setSelected] = useState(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [isFinished, setIsFinished] = useState(false);

  useEffect(() => {
    setCurrentIdx(0);
    setSelected(null);
    setShowFeedback(false);
    setIsFinished(false);
  }, [activeTab]);

  const quizzes = ALGORITHM_QUIZZES[activeTab] || [];
  if (!quizzes.length || isFinished || !quizzes[currentIdx]) {
    return isFinished ? (
      <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-2xl p-8 text-center">
        <Trophy className="mx-auto text-emerald-500 mb-4" size={48} />
        <h3 className="text-xl font-bold text-white mb-2">Quiz Completed!</h3>
        <p className="text-slate-400 text-sm mb-6">You've mastered the basics of this module.</p>
        <button 
          onClick={() => { setIsFinished(false); setCurrentIdx(0); }}
          className="px-6 py-2 bg-emerald-500 text-white rounded-lg font-medium hover:bg-emerald-600 transition-all"
        >
          Retake Quiz
        </button>
      </div>
    ) : null;
  }

  const quiz = quizzes[currentIdx];

  const handleSelect = (idx) => {
    if (showFeedback) return;
    setSelected(idx);
    setShowFeedback(true);
    
    if (idx === quiz.correct) {
      saveProgress(`${activeTab}_quiz_${currentIdx}`, 100);
    }
  };

  const handleNext = () => {
    if (currentIdx < quizzes.length - 1) {
      setCurrentIdx(currentIdx + 1);
      setSelected(null);
      setShowFeedback(false);
    } else {
      setIsFinished(true);
    }
  };

  return (
    <div className="bg-slate-900/80 border border-white/5 rounded-2xl p-6 shadow-xl">
      <div className="flex items-center gap-2 text-slate-400 text-xs font-bold uppercase tracking-widest mb-4">
        <HelpCircle size={14} className="text-brand-500" />
        <span>Quick Assessment</span>
      </div>

      <h4 className="text-lg font-bold text-white mb-6">{quiz.question}</h4>

      <div className="space-y-3">
        {quiz.options?.map((opt, i) => (
          <button
            key={i}
            onClick={() => handleSelect(i)}
            className={`w-full text-left p-4 rounded-xl border transition-all flex justify-between items-center ${
              showFeedback 
                ? i === quiz.correct 
                  ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400' 
                  : i === selected 
                    ? 'bg-ef4444/10 border-ef4444/50 text-red-400'
                    : 'bg-slate-950/50 border-white/5 text-slate-500'
                : 'bg-slate-950/50 border-white/5 text-slate-300 hover:border-brand-500/50 hover:bg-slate-900'
            }`}
          >
            <span>{opt}</span>
            {showFeedback && i === quiz.correct && <CheckCircle2 size={18} />}
            {showFeedback && i === selected && i !== quiz.correct && <XCircle size={18} />}
          </button>
        ))}
      </div>

      {showFeedback && (
        <div className="mt-6 flex flex-col gap-4">
          <p className={`text-sm ${selected === quiz.correct ? 'text-emerald-400' : 'text-red-400'}`}>
            {selected === quiz.correct ? quiz.feedback : "Not quite! Try to rethink the concept and try again."}
          </p>
          <button 
            onClick={handleNext}
            className="w-full py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl font-medium transition-all"
          >
            {currentIdx === quizzes.length - 1 ? "Finish Quiz" : "Next Question"}
          </button>
        </div>
      )}
    </div>
  );
};

export default Quiz;
