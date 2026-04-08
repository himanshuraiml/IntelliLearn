# Phased Execution Task List: IntelliLearn ML

## 🟢 Phase 1: MVP Core (Months 0–2) (COMPLETED ✅)
*Goal: Establish the base infrastructure and 3 core algorithms with interactive visuals.*

### 1.1 Project Setup & Infrastructure
- [x] Initialize Next.js project with Tailwind CSS. (MANUALLY CONFIGURED)
- [x] Configure `next.config.js` for static export.
- [x] Set up GitHub Repo & Netlify CI/CD pipeline.
- [x] Install dependencies: `tensorflow/tfjs`, `d3`, `lucide-react`.

### 1.2 Core ML Algorithm Implementation
- [x] **Linear Regression**: Implementation with stochastic gradient descent.
- [x] **Logistic Regression**: Implementation with sigmoid activation & binary cross-entropy.
- [x] **K-Nearest Neighbors (KNN)**: Custom JS implementation for distance calculation.
- [ ] **Decision Trees**: Simplified ID3/CART for 2D classification.

### 1.3 Visualization Engine
- [x] Build **ScatterPlot component** (D3) for 2D dataset rendering.
- [x] Build **Regression Line visualizer** with real-time slope/intercept updates.
- [ ] Build **Decision Boundary visualizer** using contour plots.
- [ ] Implement **Loss vs Epoch graph** using Chart.js.

### 1.4 UI & Interaction
- [x] Create **Algorithm Navigation** (Sidebar/Tabs).
- [x] Create **Parameter Slider component** (Learning rate, K, Epochs).
- [x] Implement **Real-time Training Loop** (Start/Pause/Reset).
- [x] Finalize MVP Layout (Concept -> Playground -> Results).

---

## 🟡 Phase 2: Enhanced Learning & Advanced ML (Months 2–4) (IN PROGRESS 🏗️)
*Goal: Introduce Neural Networks, Python simulation, and code transparency.*

### 2.1 Deep Learning Modules
- [x] **Perceptron**: Logical gate simulations (AND, OR, XOR).
- [x] **Basic FFNN**: Multi-layer perceptron with customizable hidden layers.
- [x] **ANN Visualizer**: Dynamic graph showing forward propagation weights.

### 2.2 Simulation & Code Integration
- [x] **Code Sync**: Auto-update a "Python/JS" code snippet box based on slider values.
- [x] **Pyodide Integration**: (Optional/Experimental) Run actual Python scripts in-browser for advanced metrics.
- [x] **Export Feature**: Allow users to download their configured model as a JSON file.

### 2.3 Pedagogy & Assessment
- [x] Implement **Guided Learning Tracks** (Sequential modules).
- [x] Create **Interactive Quizzes** with parameter-tuning challenges.
- [x] Add **LocalStorage Persistence** for user progress and quiz scores.

---

## 🔵 Phase 3: Scaling & User Experience (Months 4–8) (COMPLETED ✅)
*Goal: Polish, social features, and accessibility.*

### 3.1 Advanced UI/UX
- [x] Implement **Dark Mode** support.
- [x] Add **Animations & Transitions** for data transitions (D3 `transition()`).
- [x] Mobile optimization (Responsive charts).

### 3.2 Feature Expansion
- [x] **Multi-dataset support**: Add Iris, Titanic (simplified), and custom synth data.
- [x] **Social Sharing**: Generate unique URLs for specific model parameters.
- [x] **Community Models**: (Phase 3 Backend - Optional) Public gallery simulation.

---

## ✅ Deliverables Checklist
- [ ] `docs/`: Technical and User documentation.
- [ ] `src/`: Production-ready source code.
- [ ] `dist/`: Optimized static build assets.
- [ ] `README.md`: Setup and contribution guidelines.
