/**
 * Guided Learning Content, Quizzes, and Flashcards
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
    { title: "Max Depth & Overfitting", content: "Deeper trees fit training data better but may overfit. Try increasing max depth and watch the decision regions get more complex." },
    { title: "Feature Importance", content: "Each split is scored by how much it reduces impurity (Gini gain × samples). The chart below shows which feature the tree relied on most." }
  ],
  randomForest: [
    { title: "Ensemble Learning", content: "Random Forest trains many decision trees on random bootstrap samples of the data and averages their predictions." },
    { title: "Bootstrap Sampling", content: "Each tree sees a different random subset of the data (sampled with replacement). This diversity reduces overfitting." },
    { title: "Wisdom of the Crowd", content: "Individual trees may make mistakes, but when many vote together, errors cancel out and accuracy improves significantly." },
    { title: "Feature Importance", content: "Average Gini gain across all trees reveals which features matter most — a powerful tool for understanding your model." }
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

// ─── QUIZZES ──────────────────────────────────────────────────────────────────
export const ALGORITHM_QUIZZES = {
  linear: [
    {
      id: "q-lin-1",
      question: "Which parameter controls the step size during gradient descent?",
      options: ["Epochs", "Learning Rate", "Batch Size", "Momentum"],
      correct: 1,
      feedback: "Correct! Learning rate determines how big a step the model takes each update.",
      wrongFeedback: "Not quite. 'Epochs' is the number of full passes, not step size. Learning Rate controls the step size."
    },
    {
      id: "q-lin-2",
      question: "What does 'MSE' stand for?",
      options: ["Mean Square Error", "Model Selection Engine", "Mean Squared Error", "Minimal Step Estimation"],
      correct: 2,
      feedback: "Mean Squared Error is the standard loss for regression — it penalizes large errors more.",
      wrongFeedback: "Close but option C is the proper expansion: Mean Squared Error (not 'Mean Square Error')."
    },
    {
      id: "q-lin-3",
      question: "If the training loss stops decreasing after 50 epochs, the most likely cause is:",
      options: ["Too many epochs", "The model has converged (or the learning rate is too small)", "Overfitting", "Wrong dataset"],
      correct: 1,
      feedback: "Flat loss usually means the optimizer has found a (local) minimum. Try a higher learning rate if you suspect slow convergence.",
      wrongFeedback: "Flat loss means convergence or a stuck optimizer — not overfitting (which shows on val loss)."
    },
  ],
  poly: [
    {
      id: "q-poly-1",
      question: "What does polynomial regression do differently from linear regression?",
      options: ["Uses more data", "Adds x², x³... as features", "Uses a bigger neural network", "Changes the loss function"],
      correct: 1,
      feedback: "Feature expansion allows the model to fit curved relationships in data.",
      wrongFeedback: "The key difference is feature engineering — adding polynomial terms as extra inputs."
    },
    {
      id: "q-poly-2",
      question: "A degree-5 polynomial that fits training data perfectly but has high test error is exhibiting:",
      options: ["Underfitting", "High bias", "Overfitting", "Good generalization"],
      correct: 2,
      feedback: "Perfect training fit + poor test performance is the hallmark of overfitting. High-degree polynomials are prone to this.",
      wrongFeedback: "Perfect training, bad test = overfitting. The model memorised training noise."
    },
  ],
  ridge: [
    {
      id: "q-ridge-1",
      question: "Ridge regression adds what to the loss function?",
      options: ["L1 penalty (|w|)", "L2 penalty (w²)", "Entropy loss", "KL divergence"],
      correct: 1,
      feedback: "L2 regularization shrinks weights but keeps them non-zero, unlike L1 (Lasso).",
      wrongFeedback: "Ridge uses L2 (squared weights). L1 is Lasso regression."
    },
    {
      id: "q-ridge-2",
      question: "Increasing the ridge λ parameter will:",
      options: ["Increase model complexity", "Shrink weights toward zero more strongly", "Increase training accuracy", "Have no effect"],
      correct: 1,
      feedback: "Higher λ applies stronger regularization — weights shrink, the model becomes simpler and less likely to overfit.",
      wrongFeedback: "Larger λ means stronger penalty on weight magnitude, forcing them smaller."
    },
  ],
  logistic: [
    {
      id: "q-log-1",
      question: "What function does logistic regression use to output probabilities?",
      options: ["Linear", "Sigmoid", "Step", "ReLU"],
      correct: 1,
      feedback: "The sigmoid squeezes any value into [0,1], perfect for probability output.",
      wrongFeedback: "Logistic regression uses the sigmoid (logistic) function σ(x) = 1/(1+e^-x)."
    },
    {
      id: "q-log-2",
      question: "What is the loss function for logistic regression?",
      options: ["MSE", "MAE", "Binary Cross-Entropy", "Hinge Loss"],
      correct: 2,
      feedback: "Binary Cross-Entropy heavily penalizes confident wrong predictions.",
      wrongFeedback: "MSE is for regression. Logistic regression uses Binary Cross-Entropy."
    },
    {
      id: "q-log-3",
      question: "The decision boundary of logistic regression on 2D data is always:",
      options: ["A curve", "A straight line", "A circle", "A tree"],
      correct: 1,
      feedback: "Logistic regression is linear — its boundary is always a straight line (hyperplane in higher dims).",
      wrongFeedback: "Logistic regression is a linear model — its decision boundary is a line."
    },
  ],
  knn: [
    {
      id: "q-knn-1",
      question: "What does the 'K' in KNN stand for?",
      options: ["Kernel", "Kilo", "Number of Neighbors", "Knowledge"],
      correct: 2,
      feedback: "K is the number of nearest neighbors used for majority voting.",
      wrongFeedback: "K = number of nearest neighbors the algorithm considers when making a prediction."
    },
    {
      id: "q-knn-2",
      question: "KNN requires how much training time?",
      options: ["Lots", "Some", "None — it memorizes training data", "Depends on epochs"],
      correct: 2,
      feedback: "KNN is a lazy learner — it does no training, just stores data and searches at prediction time.",
      wrongFeedback: "KNN is lazy — it stores all training data and only does work at prediction time, not during 'training'."
    },
    {
      id: "q-knn-3",
      question: "Setting K=1 in KNN gives a boundary that is:",
      options: ["Smooth and generalizes well", "Very jagged — prone to overfitting", "Always linear", "Identical to logistic regression"],
      correct: 1,
      feedback: "K=1 memorizes every training point — extremely jagged boundary that overfits to noise.",
      wrongFeedback: "K=1 means predict by the single nearest neighbor — every training point dominates its region."
    },
  ],
  naiveBayes: [
    {
      id: "q-nb-1",
      question: "Why is Naive Bayes called 'naive'?",
      options: ["It's very simple", "It assumes feature independence", "It uses a naive optimizer", "It ignores labels"],
      correct: 1,
      feedback: "Naive Bayes assumes features are conditionally independent. Simple but often effective!",
      wrongFeedback: "The 'naive' assumption is feature independence: P(x1,x2|class) = P(x1|class)·P(x2|class)."
    },
    {
      id: "q-nb-2",
      question: "Naive Bayes works best when features are:",
      options: ["Highly correlated", "Independent (or nearly so)", "Categorical only", "Normalized to [0,1]"],
      correct: 1,
      feedback: "Since it assumes independence, NB works best when that assumption holds in the data.",
      wrongFeedback: "The independence assumption is most valid when features genuinely don't depend on each other."
    },
  ],
  svm: [
    {
      id: "q-svm-1",
      question: "What does SVM try to maximize?",
      options: ["Accuracy", "Loss", "The margin between classes", "Number of support vectors"],
      correct: 2,
      feedback: "Maximizing the margin gives better generalization to unseen data.",
      wrongFeedback: "SVM's core idea: find the hyperplane with maximum margin from both classes."
    },
    {
      id: "q-svm-2",
      question: "What is the C parameter in SVM?",
      options: ["Learning rate", "Trade-off between margin width and misclassification", "Number of support vectors", "Kernel bandwidth"],
      correct: 1,
      feedback: "High C = tight fit (small margin), low C = wide margin (allows some misclassification).",
      wrongFeedback: "C is the regularization parameter. Large C penalizes errors heavily → tight boundary."
    },
    {
      id: "q-svm-3",
      question: "Points that lie exactly on the margin boundary are called:",
      options: ["Outliers", "Centroids", "Support vectors", "Eigenvectors"],
      correct: 2,
      feedback: "Support vectors are the training points closest to the boundary — they define the margin.",
      wrongFeedback: "Only the points on the margin (support vectors) matter for the SVM solution."
    },
  ],
  decisionTree: [
    {
      id: "q-dt-1",
      question: "What metric does CART use to pick the best split?",
      options: ["Accuracy", "Gini Impurity", "Entropy", "F1 Score"],
      correct: 1,
      feedback: "CART minimizes weighted Gini impurity to find the most informative split.",
      wrongFeedback: "CART (Classification and Regression Trees) specifically uses Gini impurity."
    },
    {
      id: "q-dt-2",
      question: "A Gini impurity of 0 at a node means:",
      options: ["Equal class distribution", "All samples at that node belong to one class", "50/50 split", "Maximum uncertainty"],
      correct: 1,
      feedback: "Gini=0 is a pure node — every sample belongs to the same class. That's a perfect leaf!",
      wrongFeedback: "Gini=0 means purity — all items in the node share one label."
    },
    {
      id: "q-dt-3",
      question: "Increasing max depth will most likely:",
      options: ["Reduce training accuracy", "Increase overfitting risk", "Reduce model complexity", "Improve test accuracy always"],
      correct: 1,
      feedback: "Deeper trees fit training data more precisely — but at the cost of generalization.",
      wrongFeedback: "More depth = more splits = more complex model = higher overfitting risk."
    },
  ],
  randomForest: [
    {
      id: "q-rf-1",
      question: "Random Forest combines results from many trees using...",
      options: ["The best single tree", "Majority voting", "A neural network", "Gradient boosting"],
      correct: 1,
      feedback: "Each tree votes and the majority class wins — reducing variance and overfitting.",
      wrongFeedback: "Random Forest aggregates predictions by majority vote across all trees."
    },
    {
      id: "q-rf-2",
      question: "What is 'bootstrap sampling'?",
      options: ["Loading data from the internet", "Sampling with replacement from the training set", "Removing outliers", "Normalizing features"],
      correct: 1,
      feedback: "Each tree sees a bootstrapped sample — random rows drawn with replacement. This creates diverse trees.",
      wrongFeedback: "Bootstrap = sample with replacement. Each tree trains on a different random subset."
    },
    {
      id: "q-rf-3",
      question: "Compared to a single decision tree, Random Forest typically has:",
      options: ["Lower accuracy", "Higher variance", "Better generalization and lower variance", "The exact same boundary"],
      correct: 2,
      feedback: "Ensembling reduces variance — the forest generalizes better than any single tree.",
      wrongFeedback: "Ensemble methods reduce variance while keeping bias similar to a single tree."
    },
  ],
  kmeans: [
    {
      id: "q-km-1",
      question: "K-Means is what type of learning?",
      options: ["Supervised", "Unsupervised", "Reinforcement", "Self-supervised"],
      correct: 1,
      feedback: "K-Means clusters unlabeled data — no labels required!",
      wrongFeedback: "K-Means works on data with no labels — it's an unsupervised algorithm."
    },
    {
      id: "q-km-2",
      question: "When do K-Means centroids stop moving?",
      options: ["After K epochs", "When loss < 0.01", "When assignments no longer change", "Never"],
      correct: 2,
      feedback: "The algorithm converges when no point changes its cluster assignment.",
      wrongFeedback: "K-Means stops when the assignment step produces the same clusters as the previous iteration."
    },
    {
      id: "q-km-3",
      question: "The main weakness of K-Means is:",
      options: ["Too slow for large data", "You must specify K in advance", "It requires labels", "It can't handle 2D data"],
      correct: 1,
      feedback: "You must choose K before running — and a wrong K can produce meaningless clusters.",
      wrongFeedback: "Choosing K incorrectly is K-Means' biggest practical weakness."
    },
  ],
  nn: [
    {
      id: "q-nn-1",
      question: "What is the purpose of an activation function?",
      options: ["To load data faster", "To introduce non-linearity", "To reduce parameters", "To prevent overfitting"],
      correct: 1,
      feedback: "Without activation functions, stacking layers is equivalent to a single linear layer.",
      wrongFeedback: "Activations let networks learn non-linear patterns. Without them, depth adds no power."
    },
    {
      id: "q-nn-2",
      question: "Backpropagation computes gradients by applying the:",
      options: ["Forward pass", "Chain rule of calculus", "Softmax function", "Loss function only"],
      correct: 1,
      feedback: "Backprop uses the chain rule to compute how each weight contributed to the final loss.",
      wrongFeedback: "Backpropagation = chain rule applied layer by layer from output back to input."
    },
    {
      id: "q-nn-3",
      question: "A neural network with no hidden layers is equivalent to:",
      options: ["A random forest", "Logistic or linear regression", "K-Means", "A decision tree"],
      correct: 1,
      feedback: "Input → output with sigmoid = logistic regression. Input → output linear = linear regression.",
      wrongFeedback: "Remove all hidden layers from an MLP and you have logistic/linear regression."
    },
  ],
  dnn: [
    {
      id: "q-dnn-1",
      question: "A 'Deep' neural network typically has...",
      options: ["More data", "3+ hidden layers", "A convolutional structure", "No activation functions"],
      correct: 1,
      feedback: "Depth (many hidden layers) is what makes a neural network 'deep' and enables learning complex representations.",
      wrongFeedback: "Depth refers to the number of hidden layers — 'deep' means many layers stacked."
    },
    {
      id: "q-dnn-2",
      question: "The vanishing gradient problem makes it harder to train:",
      options: ["Shallow networks", "Very deep networks", "Decision trees", "K-Means"],
      correct: 1,
      feedback: "In deep networks, gradients shrink as they propagate back — early layers learn very slowly. ReLU helps.",
      wrongFeedback: "Vanishing gradients affect deep networks where gradients must travel through many layers."
    },
  ],
};

// ─── FLASHCARDS ───────────────────────────────────────────────────────────────
export const ALGORITHM_FLASHCARDS = {
  linear: [
    { id: 'fl-lin-1', front: 'What is the cost function for linear regression?', back: 'Mean Squared Error (MSE) = (1/n) Σ(yᵢ - ŷᵢ)². It penalises large errors quadratically.' },
    { id: 'fl-lin-2', front: 'What does gradient descent do at each step?', back: 'Computes the gradient of the loss with respect to each weight, then subtracts learning_rate × gradient from the weight.' },
    { id: 'fl-lin-3', front: 'What happens when the learning rate is too large?', back: 'The update step overshoots the minimum and loss may oscillate or diverge (exploding updates).' },
    { id: 'fl-lin-4', front: 'What is the bias term in y = wx + b?', back: 'b shifts the line vertically. Without it, the line is forced through the origin, limiting fit quality.' },
  ],
  poly: [
    { id: 'fl-poly-1', front: 'How does degree affect model complexity?', back: 'Higher degree → more flexible curve → can fit training data better, but risks overfitting on unseen data.' },
    { id: 'fl-poly-2', front: 'What is the bias-variance trade-off?', back: 'Low degree (high bias) underfits. High degree (high variance) overfits. The sweet spot minimises both.' },
  ],
  ridge: [
    { id: 'fl-ridge-1', front: 'Ridge vs. Lasso: what is the key difference?', back: 'Ridge (L2) shrinks weights toward 0 but never exactly 0. Lasso (L1) can set weights to exactly 0, performing feature selection.' },
    { id: 'fl-ridge-2', front: 'What is regularization in ML?', back: 'Adding a penalty term to the loss function that discourages overly complex models (large weights), improving generalization.' },
  ],
  logistic: [
    { id: 'fl-log-1', front: 'What is the sigmoid function?', back: 'σ(z) = 1 / (1 + e^(-z)). Maps any real number to (0, 1), making it suitable for probabilities.' },
    { id: 'fl-log-2', front: 'Why not use MSE for classification?', back: 'MSE with sigmoid creates a non-convex loss with many local minima. Cross-entropy is convex and well-suited for probability outputs.' },
    { id: 'fl-log-3', front: 'What is the decision boundary of logistic regression?', back: 'The set of points where P(y=1|x) = 0.5, i.e. where w·x + b = 0. Always a hyperplane (line in 2D).' },
  ],
  knn: [
    { id: 'fl-knn-1', front: 'What distance metric does KNN use by default?', back: 'Euclidean distance: √(Σ(xᵢ-yᵢ)²). Other options include Manhattan (L1) and Minkowski.' },
    { id: 'fl-knn-2', front: 'Is KNN parametric or non-parametric?', back: 'Non-parametric — it makes no assumptions about the data distribution and stores all training examples.' },
    { id: 'fl-knn-3', front: 'What is the time complexity of KNN prediction?', back: 'O(n·d) per query with brute-force search, where n = training points and d = dimensions. Slow for large datasets.' },
  ],
  naiveBayes: [
    { id: 'fl-nb-1', front: 'State Bayes\' theorem for classification.', back: 'P(class|x) = P(x|class) × P(class) / P(x). We pick the class that maximises the numerator.' },
    { id: 'fl-nb-2', front: 'What distribution does Gaussian Naive Bayes assume?', back: 'Each feature follows a Gaussian (normal) distribution within each class. Parameters (μ, σ²) are estimated from training data.' },
  ],
  svm: [
    { id: 'fl-svm-1', front: 'What are support vectors?', back: 'The training points closest to the decision boundary. They define the margin — removing other points would not change the boundary.' },
    { id: 'fl-svm-2', front: 'What does the C parameter trade off?', back: 'Large C → narrow margin, few misclassifications (overfits). Small C → wide margin, allows more misclassifications (underfits).' },
    { id: 'fl-svm-3', front: 'What is the kernel trick in SVM?', back: 'Implicitly maps data to a higher-dimensional space where it becomes linearly separable, without computing the transformation explicitly.' },
  ],
  decisionTree: [
    { id: 'fl-dt-1', front: 'What is Gini impurity?', back: 'Gini = 1 − Σ pᵢ². Measures class mixing at a node. 0 = pure (one class), 0.5 = maximum mixing (two equal classes).' },
    { id: 'fl-dt-2', front: 'What is pruning in decision trees?', back: 'Removing branches that provide little predictive power to reduce overfitting. Can be done during (pre-pruning) or after building (post-pruning).' },
    { id: 'fl-dt-3', front: 'How does a decision tree predict a new sample?', back: 'Follow the tree from root to a leaf by applying each split condition. The leaf\'s majority class is the prediction.' },
  ],
  randomForest: [
    { id: 'fl-rf-1', front: 'What is bagging?', back: 'Bootstrap Aggregating — train multiple models on bootstrapped samples and aggregate (average/vote) their predictions to reduce variance.' },
    { id: 'fl-rf-2', front: 'What extra randomness does Random Forest add vs. Bagging?', back: 'At each split, only a random subset of features is considered (typically √d features). This decorrelates the trees further.' },
    { id: 'fl-rf-3', front: 'What does feature importance measure in a random forest?', back: 'Average Gini gain from splits on each feature, weighted by the number of samples. Higher = more useful for classification.' },
  ],
  kmeans: [
    { id: 'fl-km-1', front: 'What is the objective function of K-Means?', back: 'Minimise the within-cluster sum of squared distances: Σ Σ ||xᵢ − μₖ||². Each centroid μₖ is the mean of its assigned points.' },
    { id: 'fl-km-2', front: 'Does K-Means guarantee finding the global optimum?', back: 'No — it converges to a local minimum. Running it multiple times with different initialisations (k-means++) helps find better solutions.' },
    { id: 'fl-km-3', front: 'How does K-Means++ improve initialisation?', back: 'Instead of random centroids, each new centroid is chosen with probability proportional to its squared distance from the nearest centroid. This spreads them out.' },
  ],
  nn: [
    { id: 'fl-nn-1', front: 'What is the role of a hidden layer?', back: 'Hidden layers learn intermediate representations — combinations of input features that make the final classification easier.' },
    { id: 'fl-nn-2', front: 'What is ReLU and why is it popular?', back: 'ReLU(x) = max(0, x). Simple, fast to compute, and avoids vanishing gradients that plagued sigmoid/tanh in deep networks.' },
    { id: 'fl-nn-3', front: 'What does the Adam optimiser improve over SGD?', back: 'Adam adapts the learning rate per-parameter using first (momentum) and second (velocity) moment estimates of gradients.' },
  ],
  dnn: [
    { id: 'fl-dnn-1', front: 'What is the vanishing gradient problem?', back: 'In deep networks, gradients shrink exponentially as they propagate back. Early layers receive tiny gradients and barely learn.' },
    { id: 'fl-dnn-2', front: 'What is batch normalisation?', back: 'Normalising activations at each layer to zero mean / unit variance within a batch. Stabilises training and allows higher learning rates.' },
    { id: 'fl-dnn-3', front: 'What is dropout and what does it prevent?', back: 'Randomly zeroing a fraction of neurons during training. Acts as regularisation — prevents co-adaptation and reduces overfitting.' },
  ],
};
