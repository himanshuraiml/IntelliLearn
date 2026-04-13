"use client";
import React, { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';

/**
 * ScatterPlot — upgraded with:
 *  - Decision boundary heatmap (showSurface prop)
 *  - Draw mode: click to add, right-click to remove data points
 *  - PCA vector arrows (regressionLine.type === 'pca')
 *
 * regressionLine shape examples:
 *  - { weight, bias }                              → linear regression line
 *  - { weights:[w1,w2], bias, type:'logistic'}     → logistic decision boundary
 *  - { weights:[w1,w2], bias, type:'svm' }         → SVM margin + boundary
 *  - { cells, xMin, xMax, yMin, yMax }             → filled decision surface
 *  - { curve: [{x,y}], type:'poly' }               → polynomial curve
 *  - { cells, …, type:'nn' }                       → NN probability surface
 *  - { centroids, type:'kmeans' }                  → cluster centroids
 *  - { type:'pca', centroid, pc1, pc2 }            → PCA eigenvectors
 *
 * Extra props:
 *  predictPoint(x, y) → { type, label, value, confidence } | null
 *  drawMode    — boolean: when true, clicks add data points
 *  drawClass   — number: class label to add (0, 1, 2, …)
 *  onPointAdded(x, y) → void
 *  onRemoveNearest(x, y) → void (remove closest existing point)
 *  showSurface — boolean (default true): whether to render decision cells
 */

const COLORS_DEFAULT    = ['#ef4444', '#0ea5e9', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
const COLORS_COLORBLIND = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE', '#EE3377'];

const ScatterPlot = ({
  data,
  regressionLine,
  isTraining,
  currentEpoch,
  totalEpochs,
  width = 800,
  height = 480,
  predictPoint,
  colorblind = false,
  showSurface = true,
  drawMode   = false,
  drawClass  = 0,
  onPointAdded,
  onRemoveNearest,
}) => {
  const COLORS = colorblind ? COLORS_COLORBLIND : COLORS_DEFAULT;

  const svgRef         = useRef();
  const gRef           = useRef(null);
  const xScaleRef      = useRef(null);
  const yScaleRef      = useRef(null);
  const predictRef     = useRef(predictPoint);
  const colorsRef      = useRef(COLORS);
  const drawModeRef    = useRef(drawMode);
  const drawClassRef   = useRef(drawClass);
  const onAddRef       = useRef(onPointAdded);
  const onRemoveRef    = useRef(onRemoveNearest);
  const iWRef          = useRef(0);
  const iHRef          = useRef(0);

  // Keep refs current without triggering re-renders
  useEffect(() => { colorsRef.current = COLORS;         }, [colorblind]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { predictRef.current   = predictPoint;  }, [predictPoint]);
  useEffect(() => { drawModeRef.current  = drawMode;      }, [drawMode]);
  useEffect(() => { drawClassRef.current = drawClass;     }, [drawClass]);
  useEffect(() => { onAddRef.current     = onPointAdded;  }, [onPointAdded]);
  useEffect(() => { onRemoveRef.current  = onRemoveNearest; }, [onRemoveNearest]);

  const margin = useMemo(() => ({ top: 24, right: 24, bottom: 44, left: 48 }), []);

  // ─── Build base chart ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!data || data.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const iW = width  - margin.left - margin.right;
    const iH = height - margin.top  - margin.bottom;
    iWRef.current = iW;
    iHRef.current = iH;

    const xExt = d3.extent(data, d => d.x);
    const yExt = d3.extent(data, d => d.y !== undefined ? d.y : d.x);
    const pad  = 1.2;

    const xScale = d3.scaleLinear().domain([xExt[0] - pad, xExt[1] + pad]).range([0, iW]);
    const yScale = d3.scaleLinear().domain([yExt[0] - pad, yExt[1] + pad]).range([iH, 0]);
    xScaleRef.current = xScale;
    yScaleRef.current = yScale;

    // Arrow marker defs for PCA vectors
    const defs = svg.append('defs');
    defs.append('clipPath').attr('id', 'plot-clip')
      .append('rect').attr('width', iW).attr('height', iH);
    ['#f59e0b', '#a78bfa'].forEach((color, i) => {
      defs.append('marker')
        .attr('id', `arrow-${i}`)
        .attr('markerWidth', 8).attr('markerHeight', 8)
        .attr('refX', 6).attr('refY', 3)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,0 L0,6 L8,3 Z')
        .attr('fill', color);
    });

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    gRef.current = g;

    // Grid
    const gridG = g.append('g').attr('class', 'grid');
    gridG.append('g').attr('class', 'grid-x')
      .call(d3.axisBottom(xScale).ticks(6).tickSize(-iH).tickFormat(''))
      .attr('transform', `translate(0,${iH})`)
      .call(ax => ax.select('.domain').remove())
      .call(ax => ax.selectAll('line').attr('stroke', '#1e293b').attr('stroke-width', 1));
    gridG.append('g').attr('class', 'grid-y')
      .call(d3.axisLeft(yScale).ticks(6).tickSize(-iW).tickFormat(''))
      .call(ax => ax.select('.domain').remove())
      .call(ax => ax.selectAll('line').attr('stroke', '#1e293b').attr('stroke-width', 1));

    // Axes
    g.append('g').attr('class', 'axis-x')
      .attr('transform', `translate(0,${iH})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .call(ax => ax.selectAll('text').attr('fill', '#94a3b8').attr('font-size', 11))
      .call(ax => ax.select('.domain').attr('stroke', '#334155'))
      .call(ax => ax.selectAll('line').attr('stroke', '#334155'));
    g.append('g').attr('class', 'axis-y')
      .call(d3.axisLeft(yScale).ticks(6))
      .call(ax => ax.selectAll('text').attr('fill', '#94a3b8').attr('font-size', 11))
      .call(ax => ax.select('.domain').attr('stroke', '#334155'))
      .call(ax => ax.selectAll('line').attr('stroke', '#334155'));

    // Layer z-order: surface → lines → points → centroids → predict → epoch-info
    g.append('g').attr('class', 'surface-layer').attr('clip-path', 'url(#plot-clip)');
    g.append('g').attr('class', 'line-layer').attr('clip-path', 'url(#plot-clip)');
    g.append('g').attr('class', 'point-layer');
    g.append('g').attr('class', 'centroid-layer');
    g.append('g').attr('class', 'predict-layer');
    g.append('g').attr('class', 'epoch-info-layer');
    g.append('g').attr('class', 'pca-layer').attr('clip-path', 'url(#plot-clip)');

    // Transparent click-capture rect
    g.append('rect')
      .attr('class', 'click-target')
      .attr('width', iW)
      .attr('height', iH)
      .attr('fill', 'transparent')
      .style('cursor', () => drawModeRef.current ? 'cell' : (predictRef.current ? 'crosshair' : 'default'))
      .on('click', function (event) {
        const [px, py] = d3.pointer(event, this);
        const dataX    = xScaleRef.current.invert(px);
        const dataY    = yScaleRef.current.invert(py);

        if (drawModeRef.current) {
          // Draw mode: add a point
          if (onAddRef.current) onAddRef.current(dataX, dataY);
          return;
        }

        // Predict mode
        const fn = predictRef.current;
        if (!fn) return;

        const result = fn(dataX, dataY);
        if (!result) return;

        const layer = gRef.current.select('.predict-layer');
        layer.selectAll('*').remove();

        const cx = xScaleRef.current(dataX);
        const cy = yScaleRef.current(dataY);
        const C  = colorsRef.current;
        const markerColor = result.type === 'regression' ? '#38bdf8' : C[result.label % C.length];

        // Ripple
        layer.append('circle').attr('cx', cx).attr('cy', cy)
          .attr('r', 6).attr('fill', 'none')
          .attr('stroke', markerColor).attr('stroke-width', 2).attr('opacity', 0.9)
          .transition().duration(600).ease(d3.easeCubicOut)
          .attr('r', 20).attr('opacity', 0);

        layer.append('circle').attr('cx', cx).attr('cy', cy)
          .attr('r', 0).attr('fill', markerColor)
          .attr('stroke', '#fff').attr('stroke-width', 2).attr('opacity', 0)
          .transition().duration(200).attr('r', 7).attr('opacity', 1);

        let labelText = '';
        if (result.type === 'regression')  labelText = `ŷ = ${result.value}`;
        else if (result.type === 'cluster') labelText = `Cluster ${result.label}`;
        else labelText = result.confidence != null
          ? `Class ${result.label}  ${(result.confidence * 100).toFixed(0)}%`
          : `Class ${result.label}`;

        const textLen  = labelText.length * 6.4 + 16;
        const tooltipX = cx + 12 + textLen > iWRef.current ? cx - textLen - 12 : cx + 12;
        const tooltipY = cy < 22 ? cy + 10 : cy - 18;

        layer.append('rect')
          .attr('x', tooltipX - 4).attr('y', tooltipY - 13)
          .attr('width', textLen).attr('height', 18).attr('rx', 4)
          .attr('fill', '#1e293b').attr('stroke', '#334155').attr('stroke-width', 1).attr('opacity', 0)
          .transition().duration(200).attr('opacity', 1);
        layer.append('text')
          .attr('x', tooltipX + 4).attr('y', tooltipY)
          .attr('fill', '#e2e8f0').attr('font-size', 11).attr('font-family', 'monospace')
          .text(labelText).attr('opacity', 0)
          .transition().duration(200).attr('opacity', 1);

        gRef.current.select('.predict-layer').raise();
      })
      .on('contextmenu', function (event) {
        event.preventDefault();
        if (!drawModeRef.current) return;
        const [px, py] = d3.pointer(event, this);
        const dataX    = xScaleRef.current.invert(px);
        const dataY    = yScaleRef.current.invert(py);
        if (onRemoveRef.current) onRemoveRef.current(dataX, dataY);
      });

    // Initial points
    g.select('.point-layer').selectAll('circle').data(data, (_, i) => i)
      .join(
        enter => enter.append('circle')
          .attr('cx', d => xScale(d.x))
          .attr('cy', d => yScale(d.y !== undefined ? d.y : d.x))
          .attr('r', 5)
          .attr('fill', d => d.label !== undefined ? COLORS[d.label % COLORS.length] : '#64748b')
          .attr('stroke', '#0f172a').attr('stroke-width', 1.5)
          .attr('opacity', 0)
          .call(e => e.transition('appear').duration(400).attr('opacity', 0.85)),
        update => update
          .attr('fill', d => d.label !== undefined ? COLORS[d.label % COLORS.length] : '#64748b')
          .transition().duration(300)
          .attr('cx', d => xScale(d.x))
          .attr('cy', d => yScale(d.y !== undefined ? d.y : d.x))
      );
  }, [data, width, height, margin, colorblind]); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Animate overlays on regressionLine / showSurface change ──────────────
  useEffect(() => {
    const g      = gRef.current;
    const xScale = xScaleRef.current;
    const yScale = yScaleRef.current;
    if (!g || !xScale || !yScale) return;

    const iW = iWRef.current;
    const iH = iHRef.current;

    const surfaceLayer   = g.select('.surface-layer');
    const lineLayer      = g.select('.line-layer');
    const centroidLayer  = g.select('.centroid-layer');
    const epochInfoLayer = g.select('.epoch-info-layer');
    const pcaLayer       = g.select('.pca-layer');

    surfaceLayer.selectAll('*').remove();
    lineLayer.selectAll('*').remove();
    centroidLayer.selectAll('*').remove();
    epochInfoLayer.selectAll('*').remove();
    pcaLayer.selectAll('*').remove();

    if (!regressionLine) {
      g.select('.predict-layer').selectAll('*').remove();
      return;
    }

    const r   = regressionLine;
    const dur = isTraining ? 60 : 500;

    // ── 1. Linear Regression Line ──────────────────────────────────────────
    if (r.weight !== undefined && !r.type) {
      const xDom = xScale.domain();
      const pts  = [
        { x: xDom[0], y: r.weight * xDom[0] + r.bias },
        { x: xDom[1], y: r.weight * xDom[1] + r.bias },
      ];
      const lineFn  = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y));
      const existing = lineLayer.selectAll('.reg-line').data([pts]);
      existing.enter().append('path').attr('class', 'reg-line')
        .attr('fill', 'none').attr('stroke', '#38bdf8').attr('stroke-width', 3)
        .attr('stroke-linecap', 'round').attr('opacity', 0).attr('d', lineFn)
        .merge(existing).transition().duration(dur).attr('d', lineFn).attr('opacity', 1);
    }

    // ── 2. Polynomial Curve ────────────────────────────────────────────────
    if (r.type === 'poly' && r.curve) {
      const curveFn = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y)).curve(d3.curveCatmullRom);
      const existing = lineLayer.selectAll('.poly-line').data([r.curve]);
      existing.enter().append('path').attr('class', 'poly-line')
        .attr('fill', 'none').attr('stroke', '#a78bfa').attr('stroke-width', 3)
        .attr('opacity', 0).attr('d', curveFn)
        .merge(existing).transition().duration(dur).attr('d', curveFn).attr('opacity', 1);
    }

    // ── 3. Logistic Decision Boundary ──────────────────────────────────────
    if (r.type === 'logistic' && r.weights) {
      const xDom  = xScale.domain();
      const [w1, w2] = r.weights;
      const pts   = [
        { x: xDom[0], y: (-w1 * xDom[0] - r.bias) / w2 },
        { x: xDom[1], y: (-w1 * xDom[1] - r.bias) / w2 },
      ];
      const lineFn  = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y));
      const existing = lineLayer.selectAll('.decision-boundary').data([pts]);
      existing.enter().append('path').attr('class', 'decision-boundary')
        .attr('fill', 'none').attr('stroke', '#f472b6')
        .attr('stroke-width', 2.5).attr('stroke-dasharray', '8,4').attr('opacity', 0).attr('d', lineFn)
        .merge(existing).transition().duration(dur).attr('d', lineFn).attr('opacity', 1);
    }

    // ── 4. SVM Boundary + Margins ──────────────────────────────────────────
    if (r.type === 'svm' && r.weights) {
      const xDom  = xScale.domain();
      const [w1, w2] = r.weights;
      const b     = r.bias;
      const norm  = Math.sqrt(w1 * w1 + w2 * w2) || 1;
      [
        { offset: 0,         dash: '0',   stroke: '#f59e0b',   w: 2.5 },
        { offset:  1 / norm, dash: '6,3', stroke: '#f59e0b55', w: 1.5 },
        { offset: -1 / norm, dash: '6,3', stroke: '#f59e0b55', w: 1.5 },
      ].forEach(({ offset, dash, stroke, w }) => {
        const pts    = xDom.map(x => ({ x, y: (-w1 * x - b + offset * norm) / w2 }));
        const lineFn = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y));
        lineLayer.append('path').attr('class', 'svm-line')
          .attr('fill', 'none').attr('stroke', stroke)
          .attr('stroke-width', w).attr('stroke-dasharray', dash)
          .attr('opacity', 0).attr('d', lineFn(pts))
          .transition().duration(dur).attr('opacity', 1);
      });
    }

    // ── 5. Decision Surface (cells) ────────────────────────────────────────
    if (r.cells && showSurface) {
      const { cells, xMin, xMax, yMin, yMax } = r;
      const resolution = Math.round(Math.sqrt(cells.length)) - 1;
      const cellW = (xScale(xMax) - xScale(xMin)) / resolution;
      const cellH = (yScale(yMin) - yScale(yMax)) / resolution;

      if (r.type === 'nn') {
        const opacityScale = d3.scaleSequential().domain([0, 1]).interpolator(t => {
          const C  = colorsRef.current;
          const c0 = d3.color(C[0]), c1 = d3.color(C[1]);
          return t < 0.5
            ? d3.interpolate(`${c0}`, '#0f172a')(1 - t * 2)
            : d3.interpolate('#0f172a', `${c1}`)((t - 0.5) * 2);
        });
        surfaceLayer.selectAll('rect.surface-cell').data(cells)
          .join('rect').attr('class', 'surface-cell')
          .attr('x', d => xScale(d.x) - cellW / 2)
          .attr('y', d => yScale(d.y) - cellH / 2)
          .attr('width', cellW + 0.5).attr('height', cellH + 0.5)
          .attr('fill', d => opacityScale(d.prob))
          .attr('opacity', 0)
          .transition().duration(dur).attr('opacity', 0.5);
      } else {
        const C = colorsRef.current;
        surfaceLayer.selectAll('rect.surface-cell').data(cells)
          .join('rect').attr('class', 'surface-cell')
          .attr('x', d => xScale(d.x) - cellW / 2)
          .attr('y', d => yScale(d.y) - cellH / 2)
          .attr('width', cellW + 0.5).attr('height', cellH + 0.5)
          .attr('fill', d => C[d.label % C.length])
          .attr('opacity', 0)
          .transition().duration(dur).attr('opacity', 0.22);
      }
    }

    // ── 6. KMeans Centroids ────────────────────────────────────────────────
    if (r.type === 'kmeans' && r.centroids) {
      const starSymbol = d3.symbol().type(d3.symbolStar).size(200);
      const C = colorsRef.current;
      centroidLayer.selectAll('.centroid').data(r.centroids)
        .join(
          enter => enter.append('path').attr('class', 'centroid')
            .attr('d', starSymbol())
            .attr('transform', d => `translate(${xScale(d.x)},${yScale(d.y)})`)
            .attr('fill', (_, i) => C[i % C.length])
            .attr('stroke', '#fff').attr('stroke-width', 1.5)
            .attr('opacity', 0)
            .call(e => e.transition().duration(300).attr('opacity', 1)),
          update => update.transition().duration(dur)
            .attr('transform', d => `translate(${xScale(d.x)},${yScale(d.y)})`).attr('opacity', 1)
        );
    }

    // ── 7. PCA Eigenvector Arrows ──────────────────────────────────────────
    if (r.type === 'pca' && r.centroid && r.pc1 && r.pc2) {
      const { centroid, pc1, pc2 } = r;
      const cx = xScale(centroid.x);
      const cy = yScale(centroid.y);

      // Centroid dot
      pcaLayer.append('circle')
        .attr('cx', cx).attr('cy', cy)
        .attr('r', 5).attr('fill', '#fff').attr('stroke', '#475569').attr('stroke-width', 1.5)
        .attr('opacity', 0).transition().duration(500).attr('opacity', 0.9);

      // PC arrows (pc.dy is negated because SVG y-axis is flipped)
      [
        { pc: pc1, color: '#f59e0b', marker: 'url(#arrow-0)', label: `PC1 ${(pc1.explained * 100).toFixed(0)}%` },
        { pc: pc2, color: '#a78bfa', marker: 'url(#arrow-1)', label: `PC2 ${(pc2.explained * 100).toFixed(0)}%` },
      ].forEach(({ pc, color, marker, label }) => {
        const tx = xScale(centroid.x + pc.dx);
        const ty = yScale(centroid.y + pc.dy);

        pcaLayer.append('line')
          .attr('x1', cx).attr('y1', cy).attr('x2', cx).attr('y2', cy)
          .attr('stroke', color).attr('stroke-width', 3)
          .attr('marker-end', marker).attr('opacity', 0)
          .transition().duration(700)
          .attr('x2', tx).attr('y2', ty).attr('opacity', 0.9);

        // Label at arrow tip
        const lx = tx + (tx > cx ? 8 : -8);
        const ly = ty + (ty > cy ? 12 : -6);
        pcaLayer.append('text')
          .attr('x', lx).attr('y', ly)
          .attr('text-anchor', tx > cx ? 'start' : 'end')
          .attr('fill', color).attr('font-size', 10).attr('font-weight', 'bold')
          .text(label).attr('opacity', 0)
          .transition().duration(700).attr('opacity', 1);
      });
    }

    // ── Epoch Label ────────────────────────────────────────────────────────
    if (currentEpoch !== undefined && totalEpochs !== undefined) {
      epochInfoLayer.append('text')
        .attr('x', iW - 8).attr('y', 8)
        .attr('text-anchor', 'end').attr('fill', '#94a3b8').attr('font-size', 11)
        .attr('font-family', 'monospace')
        .text(`Epoch ${currentEpoch + 1} / ${totalEpochs}`);
    }

    g.select('.point-layer').raise();
    g.select('.predict-layer').raise();
    g.select('.pca-layer').raise();
    g.select('.click-target').raise();
  }, [regressionLine, isTraining, currentEpoch, totalEpochs, colorblind, showSurface]); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Update point colors (K-Means / PCA projection coloring) ─────────────
  useEffect(() => {
    const g      = gRef.current;
    const xScale = xScaleRef.current;
    const yScale = yScaleRef.current;
    if (!g || !xScale || !yScale || !data) return;
    const C = colorsRef.current;
    g.select('.point-layer').selectAll('circle').data(data, (_, i) => i)
      .join(
        enter => enter.append('circle')
          .attr('cx', d => xScale(d.x))
          .attr('cy', d => yScale(d.y !== undefined ? d.y : d.x))
          .attr('r', 5)
          .attr('fill', d => d.label !== undefined ? C[d.label % C.length] : '#64748b')
          .attr('stroke', '#0f172a').attr('stroke-width', 1.5)
          .attr('opacity', 0)
          .call(e => e.transition('appear').duration(300).attr('opacity', 0.85)),
        update => update
          .transition('color').duration(200)
          .attr('cx', d => xScale(d.x))
          .attr('cy', d => yScale(d.y !== undefined ? d.y : d.x))
          .attr('fill', d => d.label !== undefined ? C[d.label % C.length] : '#64748b')
      );
  }, [data, colorblind]); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Update cursor style ──────────────────────────────────────────────────
  useEffect(() => {
    const g = gRef.current;
    if (!g) return;
    g.select('.click-target').style(
      'cursor',
      drawMode ? 'cell' : (predictPoint ? 'crosshair' : 'default')
    );
  }, [predictPoint, drawMode]);

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="max-w-full h-auto"
        style={{ background: '#0a0f1a', borderRadius: 12 }}
      />
      {isTraining && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 px-3 py-1 bg-brand-500/20 border border-brand-500/30 rounded-full text-xs text-brand-400 font-mono animate-pulse pointer-events-none">
          ⚡ Training in progress…
        </div>
      )}
      {drawMode && !isTraining && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1 bg-green-500/20 border border-green-500/30 rounded-full text-xs text-green-400 pointer-events-none">
          <span className="w-2 h-2 rounded-full inline-block" style={{ background: COLORS[drawClass % COLORS.length] }} />
          Click = add Class {drawClass} · Right-click = remove
        </div>
      )}
      {!drawMode && predictPoint && !isTraining && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 px-3 py-1 bg-slate-800/80 border border-white/10 rounded-full text-xs text-slate-400 pointer-events-none">
          Click anywhere to predict
        </div>
      )}
    </div>
  );
};

export default ScatterPlot;
