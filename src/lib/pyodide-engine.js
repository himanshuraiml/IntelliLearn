/**
 * Experimental Pyodide Integration
 * Loads Pyodide from CDN and executes Python snippets for metrics.
 */

let pyodideInstance = null;

export const loadPyodide = async () => {
  if (pyodideInstance) return pyodideInstance;

  if (typeof window === 'undefined') return null;

  // Dynamically load script if not present
  if (!window.loadPyodide) {
    const script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js";
    document.head.appendChild(script);
    
    await new Promise((resolve) => {
      script.onload = resolve;
    });
  }

  pyodideInstance = await window.loadPyodide();
  return pyodideInstance;
};

export const runPythonMetrics = async (yTrue, yPred) => {
  const py = await loadPyodide();
  if (!py) return null;

  try {
    // Inject data
    py.globals.set("y_true", yTrue);
    py.globals.set("y_pred", yPred);

    // Run Python logic (Simulating advanced metrics like F1-Score or R2)
    const result = await py.runPythonAsync(`
import json
def calculate_metrics(true_vals, pred_vals):
    # Simplified JS conversion as demo
    mse = sum((t - p) ** 2 for t, p in zip(true_vals, pred_vals)) / len(true_vals)
    return json.dumps({"python_mse": mse})

calculate_metrics(y_true.to_py(), y_pred.to_py())
    `);
    
    return JSON.parse(result);
  } catch (error) {
    console.error("Pyodide execution failed:", error);
    return null;
  }
};
