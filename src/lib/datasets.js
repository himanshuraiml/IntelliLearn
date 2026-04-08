/**
 * Dataset generators for IntelliLearn
 */

export const generateLinearData = (count = 50, slope = 2, intercept = 5, noise = 1) => {
  return Array.from({ length: count }, () => {
    const x = Math.random() * 10;
    const y = slope * x + intercept + (Math.random() - 0.5) * noise * 5;
    return { x, y };
  });
};

export const generatePolynomialData = (count = 60) => {
  return Array.from({ length: count }, () => {
    const x = Math.random() * 10 - 5;
    const y = 0.3 * x * x - 1.5 * x + 2 + (Math.random() - 0.5) * 3;
    return { x, y };
  });
};

export const generateClassificationData = (count = 60) => {
  const data = [];
  for (let i = 0; i < count / 2; i++) {
    data.push({ x: Math.random() * 4, y: Math.random() * 4, label: 0 });
    data.push({ x: Math.random() * 4 + 5, y: Math.random() * 4 + 5, label: 1 });
  }
  return data;
};

/** 3-class for multi-class algorithms */
export const generateMultiClassData = (count = 90) => {
  const centers = [
    { x: 2, y: 7, label: 0 },
    { x: 7, y: 7, label: 1 },
    { x: 4.5, y: 2, label: 2 }
  ];
  const data = [];
  for (let i = 0; i < count; i++) {
    const c = centers[i % 3];
    data.push({ x: c.x + (Math.random() - 0.5) * 3, y: c.y + (Math.random() - 0.5) * 3, label: c.label });
  }
  return data;
};

/** Linearly separable for SVM/Logistic */
export const generateLinearlySeperableData = (count = 60) => {
  const data = [];
  for (let i = 0; i < count / 2; i++) {
    data.push({ x: Math.random() * 4 + 0.5, y: Math.random() * 4 + 0.5, label: 0 });
    data.push({ x: Math.random() * 4 + 5, y: Math.random() * 4 + 5, label: 1 });
  }
  return data;
};

/** Spiral dataset for neural networks */
export const generateSpiralData = (count = 100) => {
  const data = [];
  const n = count / 2;
  for (let i = 0; i < n; i++) {
    const r = i / n * 5;
    const t = (i / n) * 4 * Math.PI + (Math.random() - 0.5) * 0.4;
    data.push({ x: r * Math.cos(t) + 5, y: r * Math.sin(t) + 5, label: 0 });
    data.push({ x: r * Math.cos(t + Math.PI) + 5, y: r * Math.sin(t + Math.PI) + 5, label: 1 });
  }
  return data;
};

/** Moon-shaped data */
export const generateMoonData = (count = 100) => {
  const data = [];
  const n = count / 2;
  for (let i = 0; i < n; i++) {
    const angle = Math.PI * (i / n);
    data.push({ x: Math.cos(angle) * 4 + 5 + (Math.random() - 0.5) * 0.5, y: Math.sin(angle) * 3 + 5 + (Math.random() - 0.5) * 0.5, label: 0 });
    data.push({ x: Math.cos(angle + Math.PI) * 4 + 5 + (Math.random() - 0.5) * 0.5, y: Math.sin(angle + Math.PI) * 3 + 3 + (Math.random() - 0.5) * 0.5, label: 1 });
  }
  return data;
};

export const generateIrisData = () => {
  const data = [];
  for (let i = 0; i < 20; i++) {
    data.push({ x: 5.1 + Math.random(), y: 3.5 + Math.random(), label: 0 }); // Setosa
    data.push({ x: 7.0 + Math.random(), y: 3.2 + Math.random(), label: 1 }); // Virginica
  }
  return data;
};

export const generateClusteringData = (count = 60, clusters = 3) => {
  const data = [];
  const centers = [{ x: 2, y: 2 }, { x: 8, y: 2 }, { x: 5, y: 8 }];
  for (let i = 0; i < count; i++) {
    const center = centers[i % Math.min(clusters, centers.length)];
    data.push({ x: center.x + (Math.random() - 0.5) * 3, y: center.y + (Math.random() - 0.5) * 3 });
  }
  return data;
};

export const generateClusteringData5 = (count = 100) => {
  const centers = [{ x: 2, y: 2 }, { x: 8, y: 2 }, { x: 5, y: 8 }, { x: 1, y: 7 }, { x: 9, y: 7 }];
  const data = [];
  for (let i = 0; i < count; i++) {
    const center = centers[i % centers.length];
    data.push({ x: center.x + (Math.random() - 0.5) * 2.5, y: center.y + (Math.random() - 0.5) * 2.5 });
  }
  return data;
};
