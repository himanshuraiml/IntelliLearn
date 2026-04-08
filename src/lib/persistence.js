/**
 * Simple hook / helper for progress persistence
 */

export const saveProgress = (topic, score) => {
  if (typeof window === 'undefined') return;
  const progress = JSON.parse(localStorage.getItem('ml_progress') || '{}');
  progress[topic] = Math.max(progress[topic] || 0, score);
  localStorage.setItem('ml_progress', JSON.stringify(progress));
};

export const getProgress = () => {
  if (typeof window === 'undefined') return {};
  return JSON.parse(localStorage.getItem('ml_progress') || '{}');
};
