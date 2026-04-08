"use client";
import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronLeft, Lightbulb } from 'lucide-react';
import { LEARNING_GUIDES } from '@/lib/learning-content';

const Guide = ({ activeTab }) => {
  const [step, setStep] = useState(0);
  const guide = LEARNING_GUIDES[activeTab] || [];

  useEffect(() => {
    setStep(0);
  }, [activeTab]);

  if (!guide.length || !guide[step]) return null;

  const currentStep = guide[step];

  return (
    <div className="bg-brand-500/10 border border-brand-500/20 rounded-2xl p-6 relative overflow-hidden">
      <div className="absolute -right-4 -top-4 opacity-10">
        <Lightbulb size={120} className="text-brand-500" />
      </div>
      
      <div className="relative z-10">
        <div className="flex items-center gap-2 text-brand-400 text-xs font-bold uppercase tracking-widest mb-2">
          <Lightbulb size={14} />
          <span>Step {step + 1} of {guide.length}</span>
        </div>
        
        <h4 className="text-lg font-bold text-white mb-2">{currentStep.title}</h4>
        <p className="text-sm text-slate-400 leading-relaxed min-h-[60px]">
          {currentStep.content}
        </p>

        <div className="flex gap-2 mt-6">
          <button 
            disabled={step === 0}
            onClick={() => setStep(step - 1)}
            className="p-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-30 rounded-lg transition-all"
          >
            <ChevronLeft size={18} />
          </button>
          <button 
            disabled={step === guide.length - 1}
            onClick={() => setStep(step + 1)}
            className="flex-1 bg-brand-500 hover:bg-brand-600 disabled:opacity-30 text-white rounded-lg font-medium flex items-center justify-center gap-2 text-sm transition-all"
          >
            Next Concept <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Guide;
