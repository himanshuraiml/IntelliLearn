/**
 * Dataset generators for IntelliLearn
 * All generators accept optional { count, noise } params.
 */

export const generateLinearData = (count = 50, noiseLevel = 1) => {
  const slope = 2, intercept = 5;
  return Array.from({ length: count }, () => {
    const x = Math.random() * 10;
    const y = slope * x + intercept + (Math.random() - 0.5) * noiseLevel * 5;
    return { x, y };
  });
};

export const generatePolynomialData = (count = 60, noiseLevel = 1) => {
  return Array.from({ length: count }, () => {
    const x = Math.random() * 10 - 5;
    const y = 0.3 * x * x - 1.5 * x + 2 + (Math.random() - 0.5) * noiseLevel * 3;
    return { x, y };
  });
};

export const generateClassificationData = (count = 60, noiseLevel = 1) => {
  const data = [];
  const spread = 1 + noiseLevel * 0.8;
  for (let i = 0; i < count / 2; i++) {
    data.push({ x: Math.random() * 4 * spread, y: Math.random() * 4 * spread, label: 0 });
    data.push({ x: Math.random() * 4 * spread + 5, y: Math.random() * 4 * spread + 5, label: 1 });
  }
  return data;
};

/** 3-class for multi-class algorithms */
export const generateMultiClassData = (count = 90, noiseLevel = 1) => {
  const centers = [
    { x: 2, y: 7, label: 0 },
    { x: 7, y: 7, label: 1 },
    { x: 4.5, y: 2, label: 2 }
  ];
  const spread = 1 + noiseLevel * 0.5;
  const data = [];
  for (let i = 0; i < count; i++) {
    const c = centers[i % 3];
    data.push({ x: c.x + (Math.random() - 0.5) * 3 * spread, y: c.y + (Math.random() - 0.5) * 3 * spread, label: c.label });
  }
  return data;
};

/** Linearly separable for SVM/Logistic */
export const generateLinearlySeperableData = (count = 60, noiseLevel = 1) => {
  const data = [];
  const spread = 0.5 + noiseLevel * 0.6;
  for (let i = 0; i < count / 2; i++) {
    data.push({ x: Math.random() * 4 * spread + 0.5, y: Math.random() * 4 * spread + 0.5, label: 0 });
    data.push({ x: Math.random() * 4 * spread + 5, y: Math.random() * 4 * spread + 5, label: 1 });
  }
  return data;
};

/** Spiral dataset for neural networks */
export const generateSpiralData = (count = 100, noiseLevel = 1) => {
  const data = [];
  const n = count / 2;
  const jitter = 0.2 + noiseLevel * 0.2;
  for (let i = 0; i < n; i++) {
    const r = i / n * 5;
    const t = (i / n) * 4 * Math.PI + (Math.random() - 0.5) * jitter;
    data.push({ x: r * Math.cos(t) + 5, y: r * Math.sin(t) + 5, label: 0 });
    data.push({ x: r * Math.cos(t + Math.PI) + 5, y: r * Math.sin(t + Math.PI) + 5, label: 1 });
  }
  return data;
};

/** Moon-shaped data */
export const generateMoonData = (count = 100, noiseLevel = 1) => {
  const data = [];
  const n = count / 2;
  const jitter = 0.3 + noiseLevel * 0.4;
  for (let i = 0; i < n; i++) {
    const angle = Math.PI * (i / n);
    data.push({ x: Math.cos(angle) * 4 + 5 + (Math.random() - 0.5) * jitter, y: Math.sin(angle) * 3 + 5 + (Math.random() - 0.5) * jitter, label: 0 });
    data.push({ x: Math.cos(angle + Math.PI) * 4 + 5 + (Math.random() - 0.5) * jitter, y: Math.sin(angle + Math.PI) * 3 + 3 + (Math.random() - 0.5) * jitter, label: 1 });
  }
  return data;
};

export const generateIrisData = (count = 40, noiseLevel = 1) => {
  const data = [];
  const jitter = 0.5 + noiseLevel * 0.5;
  const n = Math.floor(count / 2);
  for (let i = 0; i < n; i++) {
    data.push({ x: 5.1 + Math.random() * jitter, y: 3.5 + Math.random() * jitter, label: 0 }); // Setosa
    data.push({ x: 7.0 + Math.random() * jitter, y: 3.2 + Math.random() * jitter, label: 1 }); // Virginica
  }
  return data;
};

export const generateClusteringData = (count = 60, clusters = 3, noiseLevel = 1) => {
  const data = [];
  const centers = [{ x: 2, y: 2 }, { x: 8, y: 2 }, { x: 5, y: 8 }];
  const spread = 1 + noiseLevel * 0.5;
  for (let i = 0; i < count; i++) {
    const center = centers[i % Math.min(clusters, centers.length)];
    data.push({ x: center.x + (Math.random() - 0.5) * 3 * spread, y: center.y + (Math.random() - 0.5) * 3 * spread });
  }
  return data;
};

export const generateClusteringData5 = (count = 100, noiseLevel = 1) => {
  const centers = [{ x: 2, y: 2 }, { x: 8, y: 2 }, { x: 5, y: 8 }, { x: 1, y: 7 }, { x: 9, y: 7 }];
  const spread = 0.8 + noiseLevel * 0.4;
  const data = [];
  for (let i = 0; i < count; i++) {
    const center = centers[i % centers.length];
    data.push({ x: center.x + (Math.random() - 0.5) * 2.5 * spread, y: center.y + (Math.random() - 0.5) * 2.5 * spread });
  }
  return data;
};

/** Uniform blob for PCA */
export const generatePCAData = (count = 80, noiseLevel = 1) => {
  const data = [];
  const angle = Math.PI / 4;
  const scaleX = 3 + noiseLevel * 0.5;
  const scaleY = 1 + noiseLevel * 0.3;
  for (let i = 0; i < count; i++) {
    const u = (Math.random() - 0.5) * scaleX * 2;
    const v = (Math.random() - 0.5) * scaleY * 2;
    const x = u * Math.cos(angle) - v * Math.sin(angle) + 5;
    const y = u * Math.sin(angle) + v * Math.cos(angle) + 5;
    data.push({ x, y });
  }
  return data;
};
