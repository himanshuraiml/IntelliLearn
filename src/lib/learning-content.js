/**
 * Guided Learning Content & Quizzes
 */

export const LEARNING_GUIDES = {
  linear: [
    { title: "What is Linear Regression?", content: "It's the simplest ML model — it finds the 'line of best fit' through data points by minimizing the sum of squared errors." },
    { title: "Adjusting Learning Rate", content: "The 'Learning Rate' controls step size. Too high → overshoots the minimum. Too low → very slow convergence. Watch the line animate each epoch!" },
    { title: "The Training Loop", content: "Each epoch, the model sees all data and nudges the weights slightly. Watch the animated line converge as loss decreases." },
    { title: "Weight & Bias", content: "The line is y = weight·x + bias. Gradient descent adjusts both every epoch to minimize error." }
  ],
  poly: [
    { title: "Why Polynomial Regression?", content: "When data has curves, a straight line doesn't fit. Polynomial regression expands features: x → [x, x², x³...]." },
    { title: "Feature Expansion", content: "The model receives [x, x², x³…] as inputs, allowing it to fit curved patterns. Watch the curve adapt with each epoch." },
    { title: "Overfitting Risk", content: "Higher degree polynomials can overfit. They may perfectly fit training data but fail on new points." }
  ],
  ridge: [
    { title: "Ridge Regression (L2)", content: "Same as linear regression, but adds a penalty term λ·Σ(w²) to the loss. This prevents the weights from getting too large." },
    { title: "Regularization Effect", content: "L2 regularization shrinks weights toward zero but never to exactly zero. It reduces overfitting by keeping the model simpler." },
    { title: "The Lambda Parameter", content: "Higher λ → stronger regularization → simpler model. Lower λ → closer to regular linear regression." }
  ],
  logistic: [
    { title: "Classification with Probabilities", content: "Logistic regression uses a sigmoid function to output a probability (0–1) for class membership." },
    { title: "The Decision Boundary", content: "The animated dashed line separates class 0 from class 1. Points above it get label=1, below get label=0. Watch it rotate each epoch!" },
    { title: "Binary Cross-Entropy", content: "Instead of MSE, logistic regression minimizes binary cross-entropy — a loss that penalizes confident wrong predictions heavily." }
  ],
  knn: [
    { title: "Distance is Key", content: "KNN finds the K nearest neighbors of a test point and uses majority vote. It's non-parametric — no training, just memory." },
    { title: "The Decision Surface", content: "The colored background shows the predicted class at every pixel. Notice the jagged, non-linear boundaries." },
    { title: "Choosing K", content: "Small K = sensitive to noise. Large K = smoother boundaries but may miss local patterns. Try different K values!" }
  ],
  naiveBayes: [
    { title: "Bayes' Theorem", content: "P(class | features) ∝ P(class) × P(features | class). We combine class probability with likelihood of the features." },
    { title: "Independence Assumption", content: "Naive Bayes assumes all features are conditionally independent given the class — hence 'naive'. This simplifies computation." },
    { title: "Gaussian Likelihood", content: "For continuous features, we model each with a Gaussian (normal) distribution with mean and variance computed from training data." }
  ],
  svm: [
    { title: "Maximum Margin Hyperplane", content: "SVM finds the decision boundary that maximizes the margin between classes. Points closest to the boundary are called support vectors." },
    { title: "The Margin Lines", content: "The solid yellow line is the decision boundary. Dashed lines show the margin. Wider margin = better generalization." },
    { title: "C Parameter (Regularization)", content: "C trades off between maximizing margin and minimizing misclassification. High C = tight fit. Low C = wider margin, more misclassifications allowed." }
  ],
  decisionTree: [
    { title: "Splitting the Data", content: "A decision tree splits data at each node using the feature and threshold that minimizes Gini impurity." },
    { title: "Gini Impurity", content: "Gini = 1 - Σ(pᵢ²). A pure node (all one class) has Gini=0. The algorithm always picks the split that lowers Gini the most." },
    { title: "Max Depth & Overfitting", content: "Deeper trees fit training data better but may overfit. Try increasing max depth and watch the decision regions get more complex." }
  ],
  randomForest: [
    { title: "Ensemble Learning", content: "Random Forest trains many decision trees on random bootstrap samples of the data and averages their predictions." },
    { title: "Bootstrap Sampling", content: "Each tree sees a different random subset of the data (sampled with replacement). This diversity reduces overfitting." },
    { title: "Wisdom of the Crowd", content: "Individual trees may make mistakes, but when many vote together, errors cancel out and accuracy improves significantly." }
  ],
  kmeans: [
    { title: "Unsupervised Learning", content: "K-Means finds groups in unlabeled data. It places K centroid stars and iteratively assigns each point to the nearest centroid." },
    { title: "The Two Steps", content: "Step 1: Assign each point to the nearest centroid. Step 2: Move centroids to the mean of their assigned points. Repeat until stable." },
    { title: "Watch the Stars Move!", content: "The ★ symbols are centroids. Each training step, they shift toward the center of their cluster. Colors update in real-time." }
  ],
  nn: [
    { title: "Neurons & Layers", content: "A neural network stacks layers of neurons. Each neuron applies a weighted sum then an activation function." },
    { title: "ReLU Activation", content: "ReLU(x) = max(0, x). It allows the network to learn non-linear patterns and helps with vanishing gradients." },
    { title: "Decision Heatmap", content: "Unlike the single boundary of logistic regression, neural networks carve out complex, curved decision regions. Watch the heatmap evolve!" }
  ],
  dnn: [
    { title: "Why 'Deep'?", content: "A Deep Neural Network uses 3+ hidden layers. More depth allows the model to learn increasingly abstract representations of data." },
    { title: "Hierarchical Features", content: "Early layers detect simple patterns. Deeper layers combine those to recognize complex structures — like edges → shapes → objects in images." },
    { title: "Spiral Classification", content: "Try the Spiral dataset — it's nearly impossible for a shallow model. A DNN easily classifies it by learning non-linear embeddings." }
  ]
};

export const ALGORITHM_QUIZZES = {
  linear: [
    { id: "q-lin-1", question: "Which parameter controls the step size during gradient descent?", options: ["Epochs", "Learning Rate", "Batch Size", "Momentum"], correct: 1, feedback: "Correct! Learning rate determines how big a step the model takes each update." },
    { id: "q-lin-2", question: "What does 'MSE' stand for?", options: ["Mean Square Error", "Model Selection Engine", "Mean Squared Error", "Minimal Step Estimation"], correct: 2, feedback: "Mean Squared Error is the standard loss for regression — it penalizes large errors more." }
  ],
  poly: [
    { id: "q-poly-1", question: "What does polynomial regression do differently from linear regression?", options: ["Uses more data", "Adds x², x³... as features", "Uses a bigger neural network", "Changes the loss function"], correct: 1, feedback: "Feature expansion allows the model to fit curved relationships in data." }
  ],
  ridge: [
    { id: "q-ridge-1", question: "Ridge regression adds what to the loss function?", options: ["L1 penalty (|w|)", "L2 penalty (w²)", "Entropy loss", "KL divergence"], correct: 1, feedback: "L2 regularization shrinks weights but keeps them non-zero, unlike L1 (Lasso)." }
  ],
  logistic: [
    { id: "q-log-1", question: "What function does logistic regression use to output probabilities?", options: ["Linear", "Sigmoid", "Step", "ReLU"], correct: 1, feedback: "The sigmoid squeezes any value into [0,1], perfect for probability output." },
    { id: "q-log-2", question: "What is the loss function for logistic regression?", options: ["MSE", "MAE", "Binary Cross-Entropy", "Hinge Loss"], correct: 2, feedback: "Binary Cross-Entropy heavily penalizes confident wrong predictions." }
  ],
  knn: [
    { id: "q-knn-1", question: "What does the 'K' in KNN stand for?", options: ["Kernel", "Kilo", "Number of Neighbors", "Knowledge"], correct: 2, feedback: "K is the number of nearest neighbors used for majority voting." },
    { id: "q-knn-2", question: "KNN requires how much training time?", options: ["Lots", "Some", "None — it memorizes training data", "Depends on epochs"], correct: 2, feedback: "KNN is a lazy learner — it does no training, just stores data and searches at prediction time." }
  ],
  naiveBayes: [
    { id: "q-nb-1", question: "Why is Naive Bayes called 'naive'?", options: ["It's very simple", "It assumes feature independence", "It uses a naive optimizer", "It ignores labels"], correct: 1, feedback: "Naive Bayes assumes features are conditionally independent. Simple but often effective!" }
  ],
  svm: [
    { id: "q-svm-1", question: "What does SVM try to maximize?", options: ["Accuracy", "Loss", "The margin between classes", "Number of support vectors"], correct: 2, feedback: "Maximizing the margin gives better generalization to unseen data." },
    { id: "q-svm-2", question: "What is the C parameter in SVM?", options: ["Learning rate", "Trade-off between margin width and misclassification", "Number of support vectors", "Kernel bandwidth"], correct: 1, feedback: "High C = tight fit (small margin), low C = wide margin (allows some misclassification)." }
  ],
  decisionTree: [
    { id: "q-dt-1", question: "What metric does CART use to pick the best split?", options: ["Accuracy", "Gini Impurity", "Entropy", "F1 Score"], correct: 1, feedback: "CART minimizes weighted Gini impurity to find the most informative split." }
  ],
  randomForest: [
    { id: "q-rf-1", question: "Random Forest combines results from many trees using...", options: ["The best single tree", "Majority voting", "A neural network", "Gradient boosting"], correct: 1, feedback: "Each tree votes and the majority class wins — reducing variance and overfitting." }
  ],
  kmeans: [
    { id: "q-km-1", question: "K-Means is what type of learning?", options: ["Supervised", "Unsupervised", "Reinforcement", "Self-supervised"], correct: 1, feedback: "K-Means clusters unlabeled data — no labels required!" },
    { id: "q-km-2", question: "When do K-Means centroids stop moving?", options: ["After K epochs", "When loss < 0.01", "When assignments no longer change", "Never"], correct: 2, feedback: "The algorithm converges when no point changes its cluster assignment." }
  ],
  nn: [
    { id: "q-nn-1", question: "What is the purpose of an activation function?", options: ["To load data faster", "To introduce non-linearity", "To reduce parameters", "To prevent overfitting"], correct: 1, feedback: "Without activation functions, stacking layers is equivalent to a single linear layer." }
  ],
  dnn: [
    { id: "q-dnn-1", question: "A 'Deep' neural network typically has...", options: ["More data", "3+ hidden layers", "A convolutional structure", "No activation functions"], correct: 1, feedback: "Depth (many hidden layers) is what makes a neural network 'deep' and enables learning complex representations." }
  ]
};
