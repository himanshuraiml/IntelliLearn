"use client";
import React, { useEffect, useRef } from 'react';
import { X, BookOpen, CheckCircle, XCircle, Cpu } from 'lucide-react';

// ─── Math helpers (proper HTML superscript / subscript) ───────────────────────
const Sup = ({ c }) => <sup style={{ fontSize: '0.72em', verticalAlign: 'super', lineHeight: 0 }}>{c}</sup>;
const Sub = ({ c }) => <sub style={{ fontSize: '0.72em', verticalAlign: 'sub',   lineHeight: 0 }}>{c}</sub>;

// ─── Shared formula wrapper: renders one or more formula lines ────────────────
function FormulaBlock({ lines }) {
  return (
    <div className="mt-3 bg-slate-950/60 border border-white/10 rounded-xl px-5 py-4 font-mono text-sm text-brand-400 leading-[2] tracking-wide space-y-1">
      {lines.map((line, i) => <div key={i}>{line}</div>)}
    </div>
  );
}

// ─── Per-algorithm formulas ────────────────────────────────────────────────────
const FORMULAS = {
  linear: [
    <span>ŷ = Xw + b</span>,
    <span>L = (1/n) Σ (y − ŷ)<Sup c="2" /></span>,
    <span>w ← w − α · ∇L</span>,
  ],
  poly: [
    <span>ŷ = w<Sub c="0" /> + w<Sub c="1" />x + w<Sub c="2" />x<Sup c="2" /> + … + w<Sub c="d" />x<Sup c="d" /></span>,
    <span>Higher degree d → more bends, more flexible curve</span>,
  ],
  ridge: [
    <span>L = (1/n) Σ (y − ŷ)<Sup c="2" /> + λ ‖w‖<Sup c="2" /></span>,
    <span>Closed form: w = (X<Sup c="T" />X + λI)<Sup c="−1" /> X<Sup c="T" />y</span>,
    <span>λ → 0 : ordinary linear regression &nbsp;|&nbsp; λ → ∞ : all weights → 0</span>,
  ],
  logistic: [
    <span>σ(z) = 1 / (1 + e<Sup c="−z" />)</span>,
    <span>L = −(1/n) Σ [ y log ŷ + (1−y) log(1−ŷ) ]</span>,
    <span>Decision boundary: w · x + b = 0</span>,
  ],
  knn: [
    <span>d(x, q) = √ Σ (x<Sub c="i" /> − q<Sub c="i" />)<Sup c="2" /></span>,
    <span>ŷ = mode &#123; label(x<Sub c="j" />) : j ∈ K nearest neighbours &#125;</span>,
  ],
  naiveBayes: [
    <span>P(c | X) ∝ P(c) · Π P(x<Sub c="i" /> | c)</span>,
    <span>ŷ = argmax<Sub c="c" /> [ log P(c) + Σ log P(x<Sub c="i" /> | c) ]</span>,
  ],
  svm: [
    <span>Minimise: ‖w‖<Sup c="2" /> / 2 + C · Σ ξ<Sub c="i" /></span>,
    <span>Margin width = 2 / ‖w‖</span>,
    <span>Kernel: K(x<Sub c="i" />, x<Sub c="j" />) = φ(x<Sub c="i" />) · φ(x<Sub c="j" />)</span>,
  ],
  decisionTree: [
    <span>Gini = 1 − Σ p<Sub c="i" /><Sup c="2" /></span>,
    <span>Entropy = − Σ p<Sub c="i" /> log<Sub c="2" />(p<Sub c="i" />)</span>,
    <span>InfoGain = H(parent) − Σ w<Sub c="k" /> · H(child<Sub c="k" />)</span>,
  ],
  randomForest: [
    <span>ŷ = mode &#123; h<Sub c="t" />(x) : t = 1 … T &#125;</span>,
    <span>Feature importance = mean impurity decrease across all splits on that feature</span>,
  ],
  kmeans: [
    <span>Inertia = Σ<Sub c="i" /> ‖x<Sub c="i" /> − μ<Sub c="c(i)" />‖<Sup c="2" /></span>,
    <span>μ<Sub c="k" /> = (1 / |C<Sub c="k" />|) Σ<Sub c="x ∈ Cₖ" /> x</span>,
    <span>c<Sub c="i" /> = argmin<Sub c="k" /> ‖x<Sub c="i" /> − μ<Sub c="k" />‖</span>,
  ],
  nn: [
    <span>a<Sup c="l" /> = σ( W<Sup c="l" /> a<Sup c="l−1" /> + b<Sup c="l" /> )</span>,
    <span>δ<Sup c="l" /> = (W<Sup c="l+1" />)<Sup c="T" /> δ<Sup c="l+1" /> ⊙ σ′( z<Sup c="l" /> )</span>,
    <span>∂L / ∂W<Sup c="l" /> = δ<Sup c="l" /> ( a<Sup c="l−1" /> )<Sup c="T" /></span>,
  ],
  dnn: [
    <span>Vanishing gradients → ReLU activation + He initialisation</span>,
    <span>Overfitting → Dropout + L2 weight decay</span>,
    <span>Slow convergence → Adam / RMSProp + Batch Normalisation</span>,
  ],
};

// ─── Blog-style text content ───────────────────────────────────────────────────
const CONTENT = {
  linear: {
    title: 'Linear Regression',
    subtitle: 'The simplest model that fits a straight line through your data',
    color: 'from-blue-500 to-cyan-500',
    sections: [
      {
        heading: 'What is it?',
        body: `Linear regression is the cornerstone of supervised machine learning. Given a set of input features X and continuous output values y, it finds the best-fit straight line (or hyperplane in higher dimensions) by minimizing the sum of squared differences between the predicted and actual values — a technique known as Ordinary Least Squares (OLS).

The model assumes a linear relationship between inputs and outputs: ŷ = w₀ + w₁x₁ + w₂x₂ + … + wₙxₙ, where the w values (weights/coefficients) are learned during training.`,
      },
      {
        heading: 'How Training Works',
        steps: [
          'Initialize weights w randomly (or to zero).',
          'For each training step, compute the prediction ŷ = Xw.',
          'Calculate the Mean Squared Error loss: L = (1/n) Σ(yᵢ − ŷᵢ)².',
          'Compute the gradient of L with respect to w: ∇L = (2/n) Xᵀ(ŷ − y).',
          'Update weights using gradient descent: w ← w − α·∇L, where α is the learning rate.',
          'Repeat until convergence (loss stops decreasing significantly).',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'linear' },
      {
        heading: 'Pros & Cons',
        pros: [
          'Extremely fast to train — even on millions of rows.',
          'Highly interpretable — each weight tells you the effect of that feature.',
          'No hyperparameter tuning needed for basic OLS.',
          'Works perfectly when the true relationship really is linear.',
        ],
        cons: [
          'Cannot model curves or interactions between features without feature engineering.',
          'Sensitive to outliers (squared loss amplifies large errors).',
          'Assumes features are independent (multicollinearity inflates variance).',
          'Underfits complex datasets.',
        ],
      },
      {
        heading: 'When to Use',
        body: `Use linear regression when you need a fast, interpretable baseline. It excels when the relationship between input and output is genuinely linear — housing prices vs. square footage, salary vs. years of experience. Always check the residual plot: if residuals show a pattern, the relationship is non-linear and you need polynomial or tree-based models.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Predicting house prices from size, location, and age.',
          'Forecasting sales revenue from ad spend.',
          'Estimating crop yield from rainfall and temperature.',
          'Econometric modeling and policy analysis.',
        ],
      },
    ],
  },

  poly: {
    title: 'Polynomial Regression',
    subtitle: 'Capture curves by adding powers of features',
    color: 'from-violet-500 to-purple-500',
    sections: [
      {
        heading: 'What is it?',
        body: `Polynomial regression extends linear regression by adding powers of the input features (x², x³, …) as additional columns. Despite looking curved, it is still a linear model in terms of the coefficients — meaning all of gradient descent's mathematics apply unchanged.

A degree-2 model fits a parabola: ŷ = w₀ + w₁x + w₂x². A degree-3 fits a cubic curve. Each extra degree adds a new coefficient and gives the model one more "bend" to work with.`,
      },
      {
        heading: 'How Training Works',
        steps: [
          'Choose the polynomial degree d (hyperparameter).',
          'Expand each feature x into [x, x², x³, …, xᵈ] — this is feature engineering.',
          'Stack these expanded features into a design matrix X̃.',
          'Train linear regression on X̃ using gradient descent or OLS exactly as before.',
          'Evaluate on a validation set: if train loss ≪ val loss, degree is too high (overfitting).',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'poly' },
      {
        heading: 'Pros & Cons',
        pros: [
          'Can model non-linear relationships without switching to a non-linear algorithm.',
          'Still fully interpretable — each coefficient has a clear meaning.',
          'Computationally cheap for low degrees.',
          'Smooth, differentiable curve — great for physics or engineering data.',
        ],
        cons: [
          "High degree polynomials wildly oscillate between data points (Runge's phenomenon).",
          'Very sensitive to outliers at the edges of the data range.',
          'Feature expansion grows quickly: degree 5 on 10 features = many columns.',
          'Not suitable for irregular, non-smooth patterns — use tree models instead.',
        ],
      },
      {
        heading: 'When to Use',
        body: `Use polynomial regression when you can see a clear, smooth curve in a scatter plot and a linear model obviously underfits. Degree 2–3 handles most real situations. Always pair it with regularization (Ridge) for degrees above 3 to avoid wild extrapolation.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Modeling acceleration (distance ~ time²) in physics experiments.',
          'Growth curves in biology — bacterial growth, population dynamics.',
          'Dose-response curves in pharmacology.',
          'Fuel efficiency vs. speed relationships in automotive engineering.',
        ],
      },
    ],
  },

  ridge: {
    title: 'Ridge Regression',
    subtitle: 'Linear regression with a penalty to shrink large weights',
    color: 'from-teal-500 to-green-500',
    sections: [
      {
        heading: 'What is it?',
        body: `Ridge regression (also called L2 regularization or Tikhonov regularization) adds a penalty term λ·Σwᵢ² to the loss function. This extra term discourages the model from assigning very large weights to any single feature, which is the root cause of overfitting and sensitivity to multicollinearity.

The model trades a tiny increase in training error for a large reduction in variance — the bias-variance tradeoff in action. When λ = 0, Ridge is identical to ordinary linear regression. As λ → ∞, all weights shrink toward (but never quite reach) zero.`,
      },
      {
        heading: 'How Training Works',
        steps: [
          'Set regularization strength λ (the key hyperparameter).',
          'Modify the loss: L_ridge = MSE + λ·Σwᵢ².',
          'Compute the gradient of the augmented loss: ∇L_ridge = ∇MSE + 2λw.',
          'Update weights: w ← w − α·(∇MSE + 2λw).',
          'The regularization term adds a constant pull toward zero every step.',
          'Larger λ → smaller weights → simpler model. Cross-validate to choose λ.',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'ridge' },
      {
        heading: 'Pros & Cons',
        pros: [
          'Handles multicollinearity elegantly — correlated features share the load.',
          'Has a closed-form solution — no iteration needed for small datasets.',
          'Stabilizes wildly large coefficients from polynomial features.',
          'λ is easy to tune with cross-validation.',
        ],
        cons: [
          'Does not perform feature selection — all features keep non-zero weights.',
          'Requires feature scaling (standardize before applying Ridge).',
          'Interpretability slightly reduced — weights are biased by λ.',
          'If you need true sparsity, use Lasso (L1) instead.',
        ],
      },
      {
        heading: 'When to Use',
        body: `Use Ridge when you have many correlated features (e.g., genetics, NLP word counts) or when polynomial regression is overfitting. It is the preferred regularizer when you believe all features contribute something and you do not want zero-weight feature elimination.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Genomics — predicting phenotype from thousands of correlated gene expressions.',
          'Finance — portfolio optimization where assets are correlated.',
          'Image compression — regularized least-squares for signal reconstruction.',
          'NLP — regularized bag-of-words text regression.',
        ],
      },
    ],
  },

  logistic: {
    title: 'Logistic Regression',
    subtitle: 'Binary classification via a sigmoid-shaped decision boundary',
    color: 'from-orange-500 to-amber-500',
    sections: [
      {
        heading: 'What is it?',
        body: `Despite its name, logistic regression is a classification algorithm. It passes the linear combination Xw through the sigmoid function σ(z) = 1/(1 + e⁻ᶻ), squashing any real number into the range (0, 1) so it can be interpreted as a probability.

If σ(Xw) ≥ 0.5, the model predicts class 1; otherwise class 0. The decision boundary is the set of points where σ(Xw) = 0.5, i.e., where Xw = 0 — a straight line in 2D.`,
      },
      {
        heading: 'How Training Works',
        steps: [
          'Compute the linear score z = Xw + b.',
          'Apply sigmoid: ŷ = σ(z) = 1 / (1 + e⁻ᶻ).',
          'Compute Binary Cross-Entropy loss: L = −(1/n)Σ[yᵢ log ŷᵢ + (1−yᵢ) log(1−ŷᵢ)].',
          'Gradient of L w.r.t. w: ∇L = (1/n)Xᵀ(ŷ − y).',
          'Update: w ← w − α·∇L.',
          'Repeat until convergence. The loss is convex — no local minima.',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'logistic' },
      {
        heading: 'Pros & Cons',
        pros: [
          'Outputs calibrated probabilities — not just a class label.',
          'Loss function is convex — guaranteed to find the global optimum.',
          'Very fast training even on large datasets.',
          "Weights directly indicate each feature's contribution to the log-odds.",
        ],
        cons: [
          'Only creates a linear decision boundary — cannot solve XOR or spirals.',
          'Requires feature scaling for fast convergence.',
          'Can underfit complex, non-linear class boundaries.',
          'Multi-class needs One-vs-Rest or Softmax extensions.',
        ],
      },
      {
        heading: 'When to Use',
        body: `Logistic regression is your go-to first model for binary classification. It is fast, interpretable, and provides probabilities. Use it when the classes are linearly separable or nearly so. When the boundary is curved, upgrade to SVM with RBF kernel, decision trees, or neural networks.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Email spam detection — is this email spam or not?',
          'Medical diagnosis — does this patient have the disease?',
          'Credit risk — will this loan default?',
          'Click-through rate prediction in online advertising.',
        ],
      },
    ],
  },

  knn: {
    title: 'K-Nearest Neighbors',
    subtitle: 'Classify by majority vote of the K closest training points',
    color: 'from-emerald-500 to-lime-500',
    sections: [
      {
        heading: 'What is it?',
        body: `K-Nearest Neighbors (KNN) is a lazy learner — it does zero computation at training time. All the work happens at prediction time. To classify a new point, it searches the entire training set for the K closest points (by Euclidean distance, by default), then takes a majority vote of their labels.

KNN is non-parametric: it makes no assumptions about the underlying data distribution. The decision boundary is completely determined by the training data's geometry and can be arbitrarily complex.`,
      },
      {
        heading: 'How Prediction Works',
        steps: [
          'Receive a new query point q.',
          'Compute the Euclidean distance from q to every training point.',
          'Sort training points by distance, take the K nearest.',
          'Count the class labels among those K neighbors.',
          'Assign the majority class as the prediction.',
          "Ties are broken by taking the nearest neighbor's class, or by reducing K.",
        ],
      },
      { heading: 'Key Formula', formulaKey: 'knn' },
      {
        heading: 'Pros & Cons',
        pros: [
          'No training time — the model IS the data.',
          'Naturally handles multi-class problems.',
          'Works well with small datasets and complex boundaries.',
          'Immediately adapts to new training data without retraining.',
        ],
        cons: [
          'Prediction is O(n·d) — very slow on large datasets.',
          'Requires feature scaling — distance is meaningless without it.',
          'Suffers from the curse of dimensionality in high dimensions.',
          'Sensitive to irrelevant or noisy features.',
        ],
      },
      {
        heading: 'When to Use',
        body: `KNN is ideal for small-to-medium datasets where interpretability matters — you can literally point to which training examples drove a prediction. Avoid it when prediction speed is critical or when you have millions of data points. K=1 creates a very jagged boundary (overfitting); higher K smooths it out (underfitting). The sweet spot is usually found via cross-validation.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Recommendation systems — "users like you also liked…"',
          'Anomaly detection — points with no nearby neighbors are outliers.',
          'Medical image classification (with small annotated datasets).',
          'Handwriting recognition on digit datasets.',
        ],
      },
    ],
  },

  naiveBayes: {
    title: 'Naive Bayes',
    subtitle: "Probabilistic classification using Bayes' theorem and feature independence",
    color: 'from-pink-500 to-rose-500',
    sections: [
      {
        heading: 'What is it?',
        body: `Naive Bayes applies Bayes' theorem with a strong ("naive") assumption: all features are conditionally independent given the class label. Despite being almost always wrong in practice, this simplification makes the computation tractable and often surprisingly accurate.

The model computes the posterior probability P(class | features) for each class using the prior P(class) and the likelihood P(features | class), then predicts the class with the highest posterior — the Maximum A Posteriori (MAP) estimate.`,
      },
      {
        heading: 'How It Works',
        steps: [
          'Training: estimate P(class) — fraction of training points in each class.',
          'Training: for each feature, estimate P(featureᵢ | class) — e.g., mean & variance for Gaussian NB.',
          'Prediction: for each class c, compute P(c) × Π P(xᵢ | c) using the independence assumption.',
          'The Π (product) turns into a sum of log-likelihoods for numerical stability.',
          'Return the class c that maximizes the posterior probability.',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'naiveBayes' },
      {
        heading: 'Pros & Cons',
        pros: [
          'Extremely fast training and prediction — closed-form estimates.',
          'Works remarkably well with very small training sets.',
          'Handles high-dimensional data naturally (e.g., text).',
          'Naturally outputs calibrated probabilities.',
        ],
        cons: [
          'Independence assumption is almost always violated in practice.',
          'Cannot model interactions between features.',
          'Zero-frequency problem: if a feature value was never seen with a class, probability is 0 (fix with Laplace smoothing).',
          'Continuous features require distribution assumptions (Gaussian, etc.).',
        ],
      },
      {
        heading: 'When to Use',
        body: `Naive Bayes shines in text classification (spam filtering, sentiment analysis) because word occurrences are naturally high-dimensional and the bag-of-words assumption plays well with the independence assumption. It is also an excellent baseline — if NB beats your fancy model, something is wrong with the fancy model.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Spam email filtering — the classic application.',
          'Sentiment analysis of product reviews.',
          'News topic categorization.',
          'Medical diagnosis from symptom checklists.',
        ],
      },
    ],
  },

  svm: {
    title: 'Support Vector Machine',
    subtitle: 'Find the widest margin hyperplane that separates two classes',
    color: 'from-red-500 to-orange-500',
    sections: [
      {
        heading: 'What is it?',
        body: `A Support Vector Machine (SVM) finds the decision boundary (hyperplane) that maximizes the margin — the perpendicular distance from the boundary to the nearest training point on either side. Those nearest points are called support vectors; they alone define the boundary (the rest of the data is irrelevant once trained).

The soft-margin variant (C-SVM) allows some misclassifications to handle non-linearly separable data. With the kernel trick, SVMs can create non-linear boundaries by implicitly mapping data into a higher-dimensional space.`,
      },
      {
        heading: 'How Training Works',
        steps: [
          'Define the hyperplane as w·x + b = 0, with margins at w·x + b = ±1.',
          'Maximize margin width 2/‖w‖, equivalent to minimizing ‖w‖²/2.',
          'With soft margin: allow slack variables ξᵢ ≥ 0, add penalty C·Σξᵢ.',
          'Optimization problem: minimize ‖w‖²/2 + C·Σξᵢ subject to yᵢ(w·xᵢ+b) ≥ 1−ξᵢ.',
          'Solve via quadratic programming (or SMO algorithm for large datasets).',
          'Support vectors are points where the constraint is tight — they define the boundary.',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'svm' },
      {
        heading: 'Pros & Cons',
        pros: [
          'Maximum margin gives excellent generalization in high-dimensional spaces.',
          'Works well even when dimensions > samples (e.g., text, genomics).',
          'Kernel trick enables non-linear boundaries without explicit feature expansion.',
          'Only support vectors matter — efficient memory usage.',
        ],
        cons: [
          'Slow to train on large datasets (O(n²) to O(n³) complexity).',
          'Sensitive to feature scaling — always normalize before training.',
          'No probability output by default (requires Platt scaling).',
          'Choosing the right kernel and tuning C and γ requires careful cross-validation.',
        ],
      },
      {
        heading: 'When to Use',
        body: `SVMs are excellent for small-to-medium datasets where finding the best boundary matters more than training speed. They dominate in text classification and bioinformatics where features are numerous and samples are few. For large datasets (>100k points), neural networks or gradient boosted trees are more practical.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Face detection and image classification (early deep learning era).',
          'Cancer classification from gene expression data.',
          'Handwriting recognition (MNIST benchmarks).',
          'Financial fraud detection with high-dimensional transaction features.',
        ],
      },
    ],
  },

  decisionTree: {
    title: 'Decision Tree',
    subtitle: 'Recursively split data on the most informative feature threshold',
    color: 'from-yellow-500 to-amber-500',
    sections: [
      {
        heading: 'What is it?',
        body: `A decision tree learns a sequence of if-then-else rules by recursively splitting the training data. At each node, it searches all features and all possible split thresholds, choosing the split that most reduces impurity in the resulting subsets.

For classification, impurity is measured by Gini index or entropy. For regression, it minimizes variance. The tree grows until leaves are pure (one class), reach max depth, or contain fewer than min_samples_split examples.`,
      },
      {
        heading: 'How Training Works',
        steps: [
          'Start with all training data at the root node.',
          'For each feature and each possible split threshold, compute impurity reduction.',
          'Choose the (feature, threshold) pair that gives the greatest information gain.',
          'Partition the data into two child nodes: left (≤ threshold) and right (> threshold).',
          'Recurse on each child until a stopping condition is met (max depth, pure leaves, min samples).',
          'Leaf nodes store the majority class (classifier) or mean value (regressor).',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'decisionTree' },
      {
        heading: 'Pros & Cons',
        pros: [
          'Fully interpretable — you can draw the tree and explain any prediction.',
          'No feature scaling needed.',
          'Handles mixed feature types (numeric and categorical) naturally.',
          'Learns complex non-linear boundaries and feature interactions automatically.',
        ],
        cons: [
          'Tends to overfit deeply — must be pruned or depth-limited.',
          'Unstable: small changes in data → very different tree structure.',
          'Biased toward features with many possible thresholds.',
          'Not competitive with ensemble methods on standard benchmarks.',
        ],
      },
      {
        heading: 'When to Use',
        body: `Decision trees are the first choice when you need to present a model to a non-technical audience — the if/else rules are self-explanatory. Use them as building blocks for Random Forests and Gradient Boosting rather than standalone on high-stakes problems, since they tend to overfit without ensembling.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Medical decision support — triage flowcharts built from patient data.',
          'Credit approval — automated loan decisioning.',
          'Customer churn prediction with explainable rules.',
          'Fraud detection with audit-friendly reasoning.',
        ],
      },
    ],
  },

  randomForest: {
    title: 'Random Forest',
    subtitle: 'Ensemble of decorrelated trees — strong by being diverse',
    color: 'from-green-600 to-emerald-500',
    sections: [
      {
        heading: 'What is it?',
        body: `Random Forest is an ensemble method that trains many decision trees — each on a different random bootstrap sample of the data (bagging) and each considering only a random subset of features at each split (feature randomness). These two sources of randomness decorrelate the trees so that averaging their predictions reduces variance without increasing bias.

Each tree makes an independent, slightly wrong prediction. But if the errors are random and uncorrelated, they cancel out when you average many trees, yielding a much more stable and accurate model.`,
      },
      {
        heading: 'How Training Works',
        steps: [
          'Choose number of trees T and features-per-split m (typically √p for classification).',
          'For each tree t = 1 to T:',
          '  a. Draw a bootstrap sample of size n (with replacement) from training data.',
          '  b. Grow a deep decision tree, but at each split consider only m random features.',
          '  c. Do not prune — depth is controlled by the bootstrap sampling alone.',
          'Prediction: collect votes from all T trees and return the majority class (or mean for regression).',
          'Out-of-bag (OOB) samples provide a free internal validation estimate.',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'randomForest' },
      {
        heading: 'Pros & Cons',
        pros: [
          'Dramatically reduces overfitting compared to single decision trees.',
          'Built-in feature importance ranking via mean impurity decrease.',
          'Robust to outliers and noisy features.',
          'OOB error eliminates need for a separate validation set.',
        ],
        cons: [
          'Not as interpretable as a single tree — you lose the white-box property.',
          'Slower training and prediction than a single tree (scales with T).',
          'Memory-hungry — stores T complete trees.',
          'Less effective on very high-dimensional sparse data (e.g., text).',
        ],
      },
      {
        heading: 'When to Use',
        body: `Random Forest is often your safest "default" classifier for tabular data — it requires minimal preprocessing, handles mixed feature types, and rarely overfits. It is frequently the best performer in Kaggle competitions on structured data when not using gradient boosting. Always check the feature importance plot: it tells you which features actually matter.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Stock market prediction and algorithmic trading.',
          'Drug discovery — predicting molecular bioactivity.',
          'Remote sensing — land cover classification from satellite imagery.',
          'Healthcare — patient readmission risk scoring.',
        ],
      },
    ],
  },

  kmeans: {
    title: 'K-Means Clustering',
    subtitle: 'Partition unlabeled data into K groups by iterative centroid refinement',
    color: 'from-sky-500 to-blue-500',
    sections: [
      {
        heading: 'What is it?',
        body: `K-Means is an unsupervised algorithm — there are no labels. Its goal is to partition n data points into K clusters so that the total within-cluster variance (inertia) is minimized. Each cluster is represented by its centroid (geometric mean), and each point is assigned to the nearest centroid.

The algorithm alternates between two steps (E and M, as in EM algorithms) until convergence: assign points to clusters, then recompute centroids.`,
      },
      {
        heading: 'How Training Works',
        steps: [
          'Choose K (the number of clusters) — the critical hyperparameter.',
          'Initialize K centroids randomly (or with K-Means++ for better starting positions).',
          'Assignment step: assign each point to the nearest centroid by Euclidean distance.',
          'Update step: recompute each centroid as the mean of its assigned points.',
          'Repeat assignment and update until no point changes its cluster (convergence).',
          'Total cost: inertia = Σ‖xᵢ − μ_cluster(i)‖². Use the elbow plot to choose K.',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'kmeans' },
      {
        heading: 'Pros & Cons',
        pros: [
          'Very fast — O(nKd) per iteration, scales to millions of points.',
          'Simple to understand and implement.',
          'Works well when clusters are roughly spherical and equally sized.',
          'Deterministic convergence (though not to the global optimum).',
        ],
        cons: [
          'Must specify K in advance — not always obvious from the data.',
          'Sensitive to initialization — different runs can give different results.',
          'Assumes clusters are convex and isotropic (round) — fails on rings or crescents.',
          'Outliers distort centroids; use K-Medoids for robustness.',
        ],
      },
      {
        heading: 'When to Use',
        body: `K-Means is your first stop for exploratory data analysis on unlabeled datasets. Use the elbow method (plot inertia vs. K and find the "elbow") or silhouette score to choose K. When cluster shapes are irregular, try DBSCAN or Gaussian Mixture Models instead.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Customer segmentation — grouping users by purchasing behavior.',
          'Image compression — replacing each pixel color with its cluster centroid.',
          'Document clustering — grouping news articles by topic.',
          'Anomaly detection — outliers are points far from any centroid.',
        ],
      },
    ],
  },

  nn: {
    title: 'Neural Network (MLP)',
    subtitle: 'Layers of neurons that learn non-linear representations via backpropagation',
    color: 'from-fuchsia-500 to-pink-500',
    sections: [
      {
        heading: 'What is it?',
        body: `A Multi-Layer Perceptron (MLP) is a feedforward neural network with one or more hidden layers. Each neuron computes a weighted sum of its inputs, then applies a non-linear activation function (ReLU, sigmoid, tanh). Stacking layers allows the network to learn hierarchical representations: early layers detect simple patterns, deeper layers combine them into complex concepts.

The network is trained end-to-end using backpropagation: gradients of the loss are propagated backward through every layer using the chain rule, and weights are updated with gradient descent.`,
      },
      {
        heading: 'How Training Works',
        steps: [
          'Forward pass: compute activations layer by layer.',
          'Compute loss (Binary Cross-Entropy for classification, MSE for regression).',
          'Backward pass: compute ∂L/∂W for each layer using the chain rule (backpropagation).',
          'Update all weights using gradient descent, Adam, or RMSProp.',
          'Repeat for each mini-batch across all epochs.',
          'Use dropout, early stopping, or weight decay to prevent overfitting.',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'nn' },
      {
        heading: 'Pros & Cons',
        pros: [
          'Universal approximator — can learn any function given enough neurons.',
          'Automatically learns feature representations — no manual feature engineering.',
          'Scales well with data: more data → better performance.',
          'Foundation of modern deep learning (CNNs, RNNs, Transformers).',
        ],
        cons: [
          'Requires much more data than classical ML algorithms.',
          'Black box — hard to interpret learned representations.',
          'Many hyperparameters (layers, neurons, LR, batch size, optimizer, activation, dropout…).',
          'Training is computationally expensive and energy intensive.',
        ],
      },
      {
        heading: 'When to Use',
        body: `Use neural networks when you have thousands+ of training examples and when feature engineering is too expensive or impractical (images, audio, text). For small tabular datasets, gradient boosted trees often outperform neural networks with far less effort. Always start with the simplest architecture and add complexity only when needed.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Image recognition — CNNs built on MLP principles classify photos.',
          'Speech recognition — RNNs and Transformers transcribe audio.',
          'Natural language processing — ChatGPT, BERT, and all LLMs.',
          'Drug discovery — predicting protein folding (AlphaFold).',
        ],
      },
    ],
  },

  dnn: {
    title: 'Deep Neural Network',
    subtitle: 'A deeper MLP with more hidden layers for learning complex hierarchical features',
    color: 'from-indigo-500 to-violet-500',
    sections: [
      {
        heading: 'What is it?',
        body: `A Deep Neural Network (DNN) is an MLP with many hidden layers (typically 3+). Depth is the key enabler of deep learning: each additional layer can represent a more abstract, compositional feature built from the previous layer's features. A 1-layer network learns linear combinations; a 2-layer network learns curves; a deep network can learn concepts.

The challenge of deep networks is vanishing/exploding gradients — during backpropagation, gradients can shrink exponentially through layers (vanishing) or grow uncontrollably (exploding). Modern solutions include ReLU activations, batch normalization, residual connections, and careful weight initialization.`,
      },
      {
        heading: 'How Depth Helps',
        steps: [
          'Layer 1: detects low-level patterns (edges, frequencies, basic word patterns).',
          'Layer 2: combines low-level patterns into mid-level features (textures, phrases).',
          'Layer 3+: builds high-level abstractions (objects, semantics, sentence meaning).',
          'The exponential expressivity of depth means deep networks need far fewer neurons than shallow ones to represent the same function.',
          'Skip connections (ResNet) allow gradients to flow directly to early layers — enabling networks with 100+ layers.',
        ],
      },
      { heading: 'Key Formula', formulaKey: 'dnn' },
      {
        heading: 'Pros & Cons',
        pros: [
          'State-of-the-art performance on images, audio, text, and many tabular tasks.',
          'Learns hierarchical feature representations automatically.',
          'Can be fine-tuned (transfer learning) from a pre-trained model with very little data.',
          'Modular — add CNN layers for images, attention layers for sequences, etc.',
        ],
        cons: [
          'Data hungry — needs tens of thousands of examples at minimum.',
          'Computationally intensive — typically requires GPU training.',
          'Extremely many hyperparameters.',
          'Complete black box — interpretability is an active research area.',
        ],
      },
      {
        heading: 'When to Use',
        body: `Deep networks are justified when the problem has complex structure (raw pixels, audio waveforms, natural language), when you have large amounts of data, and when maximum performance outweighs interpretability. For business decisions that must be explained, prefer gradient boosting with SHAP values.`,
      },
      {
        heading: 'Real-World Applications',
        bullets: [
          'Large language models (GPT, Claude, Gemini).',
          'Self-driving car perception — classifying what is in a camera feed.',
          'Medical imaging — detecting cancer in radiology scans.',
          'Game playing AI — AlphaGo, AlphaZero, OpenAI Five.',
        ],
      },
    ],
  },
};

// ─── Section Renderer ─────────────────────────────────────────────────────────
function Section({ section, idx }) {
  return (
    <div className="mb-8">
      <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
        <span className="w-5 h-5 rounded-full bg-white/10 text-xs flex items-center justify-center text-slate-400 font-mono shrink-0">
          {idx + 1}
        </span>
        {section.heading}
      </h3>

      {section.body && (
        <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-line">{section.body}</p>
      )}

      {section.steps && (
        <ol className="space-y-2 mt-2">
          {section.steps.map((step, i) => (
            <li key={i} className="flex gap-3 text-sm text-slate-300">
              <span className="text-xs font-mono text-slate-500 mt-0.5 shrink-0 w-4">{i + 1}.</span>
              <span className="leading-relaxed">{step}</span>
            </li>
          ))}
        </ol>
      )}

      {section.formulaKey && FORMULAS[section.formulaKey] && (
        <FormulaBlock lines={FORMULAS[section.formulaKey]} />
      )}

      {section.pros && (
        <div className="grid grid-cols-1 gap-3 mt-2">
          <div>
            <div className="text-xs font-semibold text-green-400 mb-2 flex items-center gap-1.5">
              <CheckCircle size={13} /> Advantages
            </div>
            <ul className="space-y-1.5">
              {section.pros.map((p, i) => (
                <li key={i} className="text-sm text-slate-300 flex gap-2">
                  <span className="text-green-500 mt-0.5 shrink-0">+</span>{p}
                </li>
              ))}
            </ul>
          </div>
          {section.cons && (
            <div>
              <div className="text-xs font-semibold text-red-400 mb-2 flex items-center gap-1.5">
                <XCircle size={13} /> Limitations
              </div>
              <ul className="space-y-1.5">
                {section.cons.map((c, i) => (
                  <li key={i} className="text-sm text-slate-300 flex gap-2">
                    <span className="text-red-500 mt-0.5 shrink-0">−</span>{c}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {section.bullets && (
        <ul className="space-y-1.5 mt-2">
          {section.bullets.map((b, i) => (
            <li key={i} className="text-sm text-slate-300 flex gap-2">
              <span className="text-brand-400 mt-0.5 shrink-0">▸</span>{b}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ─── Main Explainer Modal ─────────────────────────────────────────────────────
export default function AlgoExplainer({ algoId, onClose }) {
  const content = CONTENT[algoId];
  const scrollRef = useRef(null);

  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }, [algoId]);

  if (!content) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="relative w-full max-w-2xl max-h-[90vh] bg-slate-900 border border-white/10 rounded-2xl shadow-2xl flex flex-col overflow-hidden">

        <div className={`h-1.5 w-full bg-gradient-to-r ${content.color} shrink-0`} />

        <div className="px-8 pt-6 pb-5 border-b border-white/5 shrink-0 flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <BookOpen size={15} className="text-slate-400" />
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Algorithm Deep Dive</span>
            </div>
            <h2 className={`text-2xl font-black bg-gradient-to-r ${content.color} bg-clip-text text-transparent`}>
              {content.title}
            </h2>
            <p className="text-sm text-slate-400 mt-1">{content.subtitle}</p>
          </div>
          <button
            onClick={onClose}
            className="shrink-0 p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-all"
            title="Close (Esc)"
          >
            <X size={18} />
          </button>
        </div>

        <div ref={scrollRef} className="overflow-y-auto flex-1 px-8 py-6">
          {content.sections.map((section, idx) => (
            <Section key={idx} section={section} idx={idx} />
          ))}
          <div className="mt-2 pt-5 border-t border-white/5 flex items-center gap-2 text-xs text-slate-600">
            <Cpu size={11} />
            <span>Interact with the playground to see this algorithm in action — adjust hyperparameters and train.</span>
          </div>
        </div>

        <div className="px-8 py-3 border-t border-white/5 bg-slate-950/40 flex items-center justify-between shrink-0">
          <span className="text-[10px] text-slate-600">Press <kbd className="bg-slate-800 px-1.5 py-0.5 rounded text-slate-400 font-mono">Esc</kbd> or click outside to close</span>
          <button
            onClick={onClose}
            className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-1.5 rounded-lg transition-all font-medium"
          >
            Back to Playground
          </button>
        </div>
      </div>
    </div>
  );
}
