"use client";
import React, { useState, useEffect } from 'react';
import { HelpCircle, CheckCircle2, XCircle, Trophy, BookOpen, RotateCcw, ChevronRight } from 'lucide-react';
import { ALGORITHM_QUIZZES, ALGORITHM_FLASHCARDS } from '@/lib/learning-content';
import { saveProgress } from '@/lib/persistence';

// ── Spaced-repetition helpers (localStorage) ─────────────────────────────────
const STORE_KEY = 'intellilearn_flashcards';

function loadFlashcardState() {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
  } catch {
    return {};
  }
}

function saveFlashcardState(state) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(state));
  } catch { /* noop */ }
}

// ── FlashCard sub-component ───────────────────────────────────────────────────
function FlashCard({ card, cardKey, onKnow, onDontKnow }) {
  const [flipped, setFlipped] = useState(false);

  useEffect(() => setFlipped(false), [card]);

  return (
    <div className="flex flex-col gap-3">
      {/* Card face */}
      <div
        onClick={() => setFlipped(v => !v)}
        className="cursor-pointer select-none min-h-[120px] rounded-xl border border-white/10 bg-slate-950/60 p-4 flex flex-col items-center justify-center text-center transition-all hover:border-brand-500/40"
        title="Click to flip"
      >
        {!flipped ? (
          <>
            <span className="text-[9px] uppercase tracking-widest text-slate-600 mb-2">Concept</span>
            <p className="text-sm font-semibold text-white leading-snug">{card.front}</p>
            <span className="text-[9px] text-slate-600 mt-3">tap to reveal</span>
          </>
        ) : (
          <>
            <span className="text-[9px] uppercase tracking-widest text-brand-400/70 mb-2">Explanation</span>
            <p className="text-xs text-slate-300 leading-relaxed">{card.back}</p>
          </>
        )}
      </div>

      {/* Response buttons — only after flip */}
      {flipped && (
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={onDontKnow}
            className="py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-semibold hover:bg-red-500/20 transition-all"
          >
            Still learning
          </button>
          <button
            onClick={onKnow}
            className="py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold hover:bg-emerald-500/20 transition-all"
          >
            Got it!
          </button>
        </div>
      )}
    </div>
  );
}

// ── Main Quiz component ───────────────────────────────────────────────────────
const Quiz = ({ activeTab }) => {
  const [mode,        setMode]        = useState('quiz');   // 'quiz' | 'flashcard'
  // Quiz state
  const [currentIdx,  setCurrentIdx]  = useState(0);
  const [selected,    setSelected]    = useState(null);
  const [showFeedback,setShowFeedback]= useState(false);
  const [isFinished,  setIsFinished]  = useState(false);
  const [score,       setScore]       = useState(0);
  // Flashcard state
  const [fcState,     setFcState]     = useState(() => loadFlashcardState());
  const [fcQueue,     setFcQueue]     = useState([]);
  const [fcIdx,       setFcIdx]       = useState(0);
  const [fcDone,      setFcDone]      = useState(false);

  const quizzes   = ALGORITHM_QUIZZES[activeTab]   || [];
  const flashcards= ALGORITHM_FLASHCARDS?.[activeTab] || [];

  // Reset everything when algorithm changes
  useEffect(() => {
    setMode('quiz');
    setCurrentIdx(0); setSelected(null); setShowFeedback(false);
    setIsFinished(false); setScore(0);
    setFcQueue(shuffleCards(flashcards));
    setFcIdx(0); setFcDone(false);
  }, [activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  // Build flashcard queue (unknown cards first, then all)
  function shuffleCards(cards) {
    const known = fcState[activeTab] || {};
    const unknown = cards.filter(c => !known[c.id]);
    const review  = cards.filter(c =>  known[c.id]);
    return [...unknown, ...review];
  }

  // ── Quiz handlers ──────────────────────────────────────────────────────────
  const handleSelect = (idx) => {
    if (showFeedback) return;
    setSelected(idx);
    setShowFeedback(true);
    const quiz = quizzes[currentIdx];
    if (idx === quiz.correct) {
      setScore(s => s + 1);
      saveProgress(`${activeTab}_quiz_${currentIdx}`, 100);
    }
  };

  const handleNext = () => {
    if (currentIdx < quizzes.length - 1) {
      setCurrentIdx(currentIdx + 1);
      setSelected(null); setShowFeedback(false);
    } else {
      setIsFinished(true);
    }
  };

  // ── Flashcard handlers ─────────────────────────────────────────────────────
  const markKnown = () => {
    const card = fcQueue[fcIdx];
    const next = { ...fcState, [activeTab]: { ...(fcState[activeTab] || {}), [card.id]: true } };
    setFcState(next);
    saveFlashcardState(next);
    advanceFc();
  };

  const markUnknown = () => { advanceFc(); };

  const advanceFc = () => {
    if (fcIdx + 1 >= fcQueue.length) {
      setFcDone(true);
    } else {
      setFcIdx(f => f + 1);
    }
  };

  const resetFc = () => {
    const cleared = { ...fcState };
    delete cleared[activeTab];
    setFcState(cleared);
    saveFlashcardState(cleared);
    setFcQueue(shuffleCards(flashcards));
    setFcIdx(0); setFcDone(false);
  };

  const knownCount = Object.keys(fcState[activeTab] || {}).length;

  // ── Render: Quiz ───────────────────────────────────────────────────────────
  const renderQuiz = () => {
    if (!quizzes.length) return null;

    if (isFinished) {
      return (
        <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-2xl p-6 text-center">
          <Trophy className="mx-auto text-emerald-500 mb-3" size={40} />
          <h3 className="text-lg font-bold text-white mb-1">Quiz Completed!</h3>
          <p className="text-slate-400 text-sm mb-1">
            Score: <span className="text-emerald-400 font-bold">{score} / {quizzes.length}</span>
          </p>
          <p className="text-slate-500 text-xs mb-4">
            {score === quizzes.length ? 'Perfect! Try the flashcards next.' : 'Review the flashcards to reinforce concepts.'}
          </p>
          <div className="flex gap-2 justify-center">
            <button onClick={() => { setIsFinished(false); setCurrentIdx(0); setScore(0); setSelected(null); setShowFeedback(false); }}
              className="px-4 py-2 bg-slate-800 text-white rounded-lg text-xs font-medium hover:bg-slate-700 transition-all flex items-center gap-1.5">
              <RotateCcw size={12} /> Retake
            </button>
            {flashcards.length > 0 && (
              <button onClick={() => setMode('flashcard')}
                className="px-4 py-2 bg-brand-500 text-white rounded-lg text-xs font-medium hover:bg-brand-600 transition-all flex items-center gap-1.5">
                <BookOpen size={12} /> Flashcards
              </button>
            )}
          </div>
        </div>
      );
    }

    const quiz = quizzes[currentIdx];
    if (!quiz) return null;

    return (
      <>
        <h4 className="text-sm font-bold text-white mb-4 leading-snug">{quiz.question}</h4>

        <div className="space-y-2">
          {quiz.options?.map((opt, i) => (
            <button key={i} onClick={() => handleSelect(i)}
              className={`w-full text-left p-3 rounded-xl border transition-all flex justify-between items-center text-xs ${
                showFeedback
                  ? i === quiz.correct
                    ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400'
                    : i === selected
                      ? 'bg-red-500/10 border-red-500/50 text-red-400'
                      : 'bg-slate-950/50 border-white/5 text-slate-500'
                  : 'bg-slate-950/50 border-white/5 text-slate-300 hover:border-brand-500/50 hover:bg-slate-900'
              }`}>
              <span>{opt}</span>
              {showFeedback && i === quiz.correct && <CheckCircle2 size={15} />}
              {showFeedback && i === selected && i !== quiz.correct && <XCircle size={15} />}
            </button>
          ))}
        </div>

        {showFeedback && (
          <div className="mt-4 flex flex-col gap-3">
            <p className={`text-xs leading-relaxed ${selected === quiz.correct ? 'text-emerald-400' : 'text-red-400'}`}>
              {selected === quiz.correct ? quiz.feedback : quiz.wrongFeedback || 'Not quite! Try to rethink the concept.'}
            </p>
            <button onClick={handleNext}
              className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-medium transition-all flex items-center justify-center gap-1.5">
              {currentIdx === quizzes.length - 1 ? 'Finish Quiz' : 'Next Question'} <ChevronRight size={13} />
            </button>
          </div>
        )}
      </>
    );
  };

  // ── Render: Flashcards ─────────────────────────────────────────────────────
  const renderFlashcards = () => {
    if (!flashcards.length) {
      return <p className="text-xs text-slate-500 text-center py-4">No flashcards for this algorithm yet.</p>;
    }

    if (fcDone) {
      return (
        <div className="text-center py-4">
          <Trophy className="mx-auto text-brand-500 mb-2" size={36} />
          <p className="text-sm font-bold text-white mb-1">Deck complete!</p>
          <p className="text-xs text-slate-400 mb-1">
            Known: <span className="text-emerald-400 font-bold">{knownCount}</span> / {flashcards.length}
          </p>
          {knownCount < flashcards.length && (
            <p className="text-[10px] text-slate-500 mb-3">Repeat the deck to master remaining cards.</p>
          )}
          <div className="flex gap-2 justify-center">
            <button onClick={resetFc}
              className="px-4 py-2 bg-slate-800 text-white rounded-lg text-xs font-medium hover:bg-slate-700 flex items-center gap-1.5">
              <RotateCcw size={12} /> Reset progress
            </button>
            <button onClick={() => { setFcQueue(shuffleCards(flashcards)); setFcIdx(0); setFcDone(false); }}
              className="px-4 py-2 bg-brand-500 text-white rounded-lg text-xs font-medium hover:bg-brand-600 flex items-center gap-1.5">
              Review again
            </button>
          </div>
        </div>
      );
    }

    const card = fcQueue[fcIdx];
    if (!card) return null;

    return (
      <>
        <div className="flex justify-between text-[10px] text-slate-600 mb-3">
          <span>{fcIdx + 1} / {fcQueue.length}</span>
          <span>Known: {knownCount}/{flashcards.length}</span>
        </div>
        <div className="w-full h-1 bg-slate-800 rounded-full mb-4">
          <div className="h-full bg-brand-500 rounded-full transition-all" style={{ width: `${(fcIdx / fcQueue.length) * 100}%` }} />
        </div>
        <FlashCard card={card} cardKey={`${activeTab}-${fcIdx}`} onKnow={markKnown} onDontKnow={markUnknown} />
      </>
    );
  };

  if (!quizzes.length && !flashcards.length) return null;

  return (
    <div className="bg-slate-900/80 border border-white/5 rounded-2xl p-5 shadow-xl">
      {/* Header + mode toggle */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-400">
          <HelpCircle size={13} className="text-brand-500" />
          <span>{mode === 'quiz' ? 'Quick Assessment' : 'Flashcards'}</span>
          {mode === 'quiz' && quizzes.length > 0 && (
            <span className="text-slate-600 font-normal normal-case">
              {currentIdx + 1}/{quizzes.length}
            </span>
          )}
        </div>

        {/* Toggle buttons */}
        <div className="flex rounded-lg overflow-hidden border border-white/10 text-[10px]">
          <button
            onClick={() => setMode('quiz')}
            className={`px-2.5 py-1 font-medium transition-all ${mode === 'quiz' ? 'bg-brand-500/20 text-brand-400' : 'text-slate-500 hover:text-slate-300'}`}
          >
            Quiz
          </button>
          {flashcards.length > 0 && (
            <button
              onClick={() => setMode('flashcard')}
              className={`px-2.5 py-1 font-medium transition-all border-l border-white/10 ${mode === 'flashcard' ? 'bg-brand-500/20 text-brand-400' : 'text-slate-500 hover:text-slate-300'}`}
            >
              Cards
            </button>
          )}
        </div>
      </div>

      {mode === 'quiz' ? renderQuiz() : renderFlashcards()}
    </div>
  );
};

export default Quiz;
