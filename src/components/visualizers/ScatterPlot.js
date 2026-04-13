"use client";
import React, { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';

/**
 * ScatterPlot with animated epoch-by-epoch training visualization + click-to-predict.
 *
 * regressionLine shape examples:
 *  - { weight, bias }                              → linear regression line
 *  - { weights:[w1,w2], bias, type:'logistic'}     → logistic decision boundary
 *  - { weights:[w1,w2], bias, type:'svm' }         → SVM margin + boundary
 *  - { cells, xMin, xMax, yMin, yMax }             → filled decision surface
 *  - { curve: [{x,y}], type:'poly' }               → polynomial curve
 *  - { cells, …, type:'nn' }                       → NN probability surface
 *  - { centroids, type:'kmeans' }                  → cluster centroids
 *
 * Extra props:
 *  predictPoint(x, y) → { type, label, value, confidence } | null
 *    When provided, clicking the plot shows an animated prediction marker.
 */

const COLORS_DEFAULT    = ['#ef4444', '#0ea5e9', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
// Blue/orange colorblind-safe palette (avoids red/green confusion)
const COLORS_COLORBLIND = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE', '#EE3377'];

const ScatterPlot = ({
  data,
  regressionLine,
  isTraining,
  currentEpoch,
  totalEpochs,
  width = 800,
  height = 480,
  predictPoint,        // optional (x,y)=>result
  colorblind = false,  // use colorblind-friendly palette
}) => {
  const COLORS = colorblind ? COLORS_COLORBLIND : COLORS_DEFAULT;
  const svgRef        = useRef();
  const gRef          = useRef(null);
  const xScaleRef     = useRef(null);
  const yScaleRef     = useRef(null);
  const predictRef    = useRef(predictPoint);
  const colorsRef     = useRef(COLORS);
  const iWRef         = useRef(0);
  const iHRef         = useRef(0);

  // Keep colorsRef fresh when palette changes
  useEffect(() => { colorsRef.current = COLORS; }, [colorblind]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keep predictRef fresh without rebuilding the base chart
  useEffect(() => { predictRef.current = predictPoint; }, [predictPoint]);

  const margin = useMemo(() => ({ top: 24, right: 24, bottom: 44, left: 48 }), []);

  // ─── Build base chart whenever data / dimensions change ────────────────────
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

    const defs = svg.append('defs');
    defs.append('clipPath').attr('id', 'plot-clip')
      .append('rect').attr('width', iW).attr('height', iH);

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

    // Layer groups (z-order: surface < lines < points < centroids < predict < epoch-info)
    g.append('g').attr('class', 'surface-layer').attr('clip-path', 'url(#plot-clip)');
    g.append('g').attr('class', 'line-layer').attr('clip-path', 'url(#plot-clip)');
    g.append('g').attr('class', 'point-layer');
    g.append('g').attr('class', 'centroid-layer');
    g.append('g').attr('class', 'predict-layer');
    g.append('g').attr('class', 'epoch-info-layer');

    // Transparent click-capture rect (on top of all layers except predict/epoch)
    g.append('rect')
      .attr('class', 'click-target')
      .attr('width', iW)
      .attr('height', iH)
      .attr('fill', 'transparent')
      .style('cursor', () => predictRef.current ? 'crosshair' : 'default')
      .on('click', function (event) {
        const fn = predictRef.current;
        if (!fn) return;

        const [px, py] = d3.pointer(event, this);
        const dataX    = xScaleRef.current.invert(px);
        const dataY    = yScaleRef.current.invert(py);

        const result = fn(dataX, dataY);
        if (!result) return;

        const layer = gRef.current.select('.predict-layer');
        layer.selectAll('*').remove();

        const cx = xScaleRef.current(dataX);
        const cy = yScaleRef.current(dataY);

        const C = colorsRef.current;
        const markerColor = result.type === 'regression'
          ? '#38bdf8'
          : C[result.label % C.length];

        // Ripple ring
        layer.append('circle')
          .attr('cx', cx).attr('cy', cy)
          .attr('r', 6).attr('fill', 'none')
          .attr('stroke', markerColor).attr('stroke-width', 2).attr('opacity', 0.9)
          .transition().duration(600).ease(d3.easeCubicOut)
          .attr('r', 20).attr('opacity', 0);

        // Solid dot
        layer.append('circle')
          .attr('cx', cx).attr('cy', cy)
          .attr('r', 0)
          .attr('fill', markerColor)
          .attr('stroke', '#fff').attr('stroke-width', 2)
          .attr('opacity', 0)
          .transition().duration(200)
          .attr('r', 7).attr('opacity', 1);

        // Label text
        let labelText = '';
        if (result.type === 'regression') {
          labelText = `ŷ = ${result.value}`;
        } else if (result.type === 'cluster') {
          labelText = `Cluster ${result.label}`;
        } else {
          labelText = result.confidence != null
            ? `Class ${result.label}  ${(result.confidence * 100).toFixed(0)}%`
            : `Class ${result.label}`;
        }

        const textLen   = labelText.length * 6.4 + 16;
        const tooltipX  = cx + 12 + textLen > iWRef.current ? cx - textLen - 12 : cx + 12;
        const tooltipY  = cy < 22 ? cy + 10 : cy - 18;

        layer.append('rect')
          .attr('x', tooltipX - 4).attr('y', tooltipY - 13)
          .attr('width', textLen).attr('height', 18)
          .attr('rx', 4)
          .attr('fill', '#1e293b').attr('stroke', '#334155').attr('stroke-width', 1)
          .attr('opacity', 0)
          .transition().duration(200).attr('opacity', 1);

        layer.append('text')
          .attr('x', tooltipX + 4).attr('y', tooltipY)
          .attr('fill', '#e2e8f0').attr('font-size', 11).attr('font-family', 'monospace')
          .text(labelText)
          .attr('opacity', 0)
          .transition().duration(200).attr('opacity', 1);

        // Raise predict layer so it's visible above everything
        gRef.current.select('.predict-layer').raise();
      });

    // Points (initial draw)
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

  // ─── Animate overlays on each regressionLine update ────────────────────────
  useEffect(() => {
    const g      = gRef.current;
    const xScale = xScaleRef.current;
    const yScale = yScaleRef.current;
    if (!g || !xScale || !yScale) return;

    const iW = iWRef.current;
    const iH = iHRef.current;

    const surfaceLayer    = g.select('.surface-layer');
    const lineLayer       = g.select('.line-layer');
    const centroidLayer   = g.select('.centroid-layer');
    const epochInfoLayer  = g.select('.epoch-info-layer');

    surfaceLayer.selectAll('*').remove();
    lineLayer.selectAll('*').remove();
    centroidLayer.selectAll('*').remove();
    epochInfoLayer.selectAll('*').remove();

    // Also clear old prediction marker when model is reset
    if (!regressionLine) {
      g.select('.predict-layer').selectAll('*').remove();
      return;
    }

    const r             = regressionLine;
    const transitionDur = isTraining ? 60 : 500;

    // ── 1. Linear Regression Line ──────────────────
    if (r.weight !== undefined && !r.type) {
      const xDom  = xScale.domain();
      const pts   = [
        { x: xDom[0], y: r.weight * xDom[0] + r.bias },
        { x: xDom[1], y: r.weight * xDom[1] + r.bias },
      ];
      const lineFn  = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y));
      const existing = lineLayer.selectAll('.reg-line').data([pts]);
      existing.enter()
        .append('path').attr('class', 'reg-line')
        .attr('fill', 'none').attr('stroke', '#38bdf8').attr('stroke-width', 3)
        .attr('stroke-linecap', 'round').attr('opacity', 0).attr('d', lineFn)
        .merge(existing)
        .transition().duration(transitionDur)
        .attr('d', lineFn).attr('opacity', 1);
    }

    // ── 2. Polynomial Regression Curve ─────────────
    if (r.type === 'poly' && r.curve) {
      const curveFn  = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y)).curve(d3.curveCatmullRom);
      const existing = lineLayer.selectAll('.poly-line').data([r.curve]);
      existing.enter()
        .append('path').attr('class', 'poly-line')
        .attr('fill', 'none').attr('stroke', '#a78bfa').attr('stroke-width', 3)
        .attr('opacity', 0).attr('d', curveFn)
        .merge(existing)
        .transition().duration(transitionDur)
        .attr('d', curveFn).attr('opacity', 1);
    }

    // ── 3. Logistic Decision Boundary ──────────────
    if (r.type === 'logistic' && r.weights) {
      const xDom  = xScale.domain();
      const [w1, w2] = r.weights;
      const pts   = [
        { x: xDom[0], y: (-w1 * xDom[0] - r.bias) / w2 },
        { x: xDom[1], y: (-w1 * xDom[1] - r.bias) / w2 },
      ];
      const lineFn  = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y));
      const existing = lineLayer.selectAll('.decision-boundary').data([pts]);
      existing.enter()
        .append('path').attr('class', 'decision-boundary')
        .attr('fill', 'none').attr('stroke', '#f472b6')
        .attr('stroke-width', 2.5).attr('stroke-dasharray', '8,4').attr('opacity', 0).attr('d', lineFn)
        .merge(existing)
        .transition().duration(transitionDur)
        .attr('d', lineFn).attr('opacity', 1);
    }

    // ── 4. SVM Boundary + Margins ──────────────────
    if (r.type === 'svm' && r.weights) {
      const xDom  = xScale.domain();
      const [w1, w2] = r.weights;
      const b     = r.bias;
      const norm  = Math.sqrt(w1 * w1 + w2 * w2) || 1;
      const offsets    = [0,       1 / norm,    -1 / norm];
      const dashArrays = ['0',     '6,3',       '6,3'];
      const strokes    = ['#f59e0b', '#f59e0b55', '#f59e0b55'];
      const widths     = [2.5, 1.5, 1.5];
      offsets.forEach((offset, oi) => {
        const pts   = xDom.map(x => ({ x, y: (-w1 * x - b + offset * norm) / w2 }));
        const lineFn = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y));
        lineLayer.append('path').attr('class', 'svm-line')
          .attr('fill', 'none').attr('stroke', strokes[oi])
          .attr('stroke-width', widths[oi]).attr('stroke-dasharray', dashArrays[oi])
          .attr('opacity', 0).attr('d', lineFn(pts))
          .transition().duration(transitionDur).attr('opacity', 1);
      });
    }

    // ── 5. Decision Surface (KNN / NB / DTree / RF / NN) ───────────────────
    if (r.cells) {
      const { cells, xMin, xMax, yMin, yMax } = r;
      const resolution = Math.round(Math.sqrt(cells.length)) - 1;
      const cellW = (xScale(xMax) - xScale(xMin)) / resolution;
      const cellH = (yScale(yMin) - yScale(yMax)) / resolution;

      if (r.type === 'nn') {
        const opacityScale = d3.scaleSequential()
          .domain([0, 1]).interpolator(t => {
            const c0 = d3.color(COLORS[0]), c1 = d3.color(COLORS[1]);
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
          .transition().duration(transitionDur).attr('opacity', 0.45);
      } else {
        surfaceLayer.selectAll('rect.surface-cell').data(cells)
          .join('rect').attr('class', 'surface-cell')
          .attr('x', d => xScale(d.x) - cellW / 2)
          .attr('y', d => yScale(d.y) - cellH / 2)
          .attr('width', cellW + 0.5).attr('height', cellH + 0.5)
          .attr('fill', d => COLORS[d.label % COLORS.length])
          .attr('opacity', 0)
          .transition().duration(transitionDur).attr('opacity', 0.2);
      }
    }

    // ── 6. KMeans Centroids ────────────────────────
    if (r.type === 'kmeans' && r.centroids) {
      const starSymbol = d3.symbol().type(d3.symbolStar).size(200);
      centroidLayer.selectAll('.centroid').data(r.centroids)
        .join(
          enter => enter.append('path').attr('class', 'centroid')
            .attr('d', starSymbol())
            .attr('transform', d => `translate(${xScale(d.x)},${yScale(d.y)})`)
            .attr('fill', (_, i) => COLORS[i % COLORS.length])
            .attr('stroke', '#fff').attr('stroke-width', 1.5)
            .attr('opacity', 0)
            .call(e => e.transition().duration(300).attr('opacity', 1)),
          update => update.transition().duration(transitionDur)
            .attr('transform', d => `translate(${xScale(d.x)},${yScale(d.y)})`).attr('opacity', 1)
        );
    }

    // ── Epoch Label ───────────────────────────────
    if (currentEpoch !== undefined && totalEpochs !== undefined) {
      epochInfoLayer.append('text')
        .attr('x', iW - 8).attr('y', 8)
        .attr('text-anchor', 'end').attr('fill', '#94a3b8').attr('font-size', 11)
        .attr('font-family', 'monospace')
        .text(`Epoch ${currentEpoch + 1} / ${totalEpochs}`);
    }

    g.select('.point-layer').raise();
    g.select('.predict-layer').raise();
    g.select('.click-target').raise();

  }, [regressionLine, isTraining, currentEpoch, totalEpochs, colorblind]); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Update point colors when K-Means reassigns clusters ──────────────────
  useEffect(() => {
    const g      = gRef.current;
    const xScale = xScaleRef.current;
    const yScale = yScaleRef.current;
    if (!g || !xScale || !yScale || !data) return;
    g.select('.point-layer').selectAll('circle').data(data, (_, i) => i)
      .transition('color').duration(200)
      .attr('fill', d => d.label !== undefined ? COLORS[d.label % COLORS.length] : '#64748b');
  }, [data, colorblind]); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Update cursor style when predictPoint changes ─────────────────────────
  useEffect(() => {
    const g = gRef.current;
    if (!g) return;
    g.select('.click-target').style('cursor', predictPoint ? 'crosshair' : 'default');
  }, [predictPoint]);

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
      {predictPoint && !isTraining && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 px-3 py-1 bg-slate-800/80 border border-white/10 rounded-full text-xs text-slate-400 pointer-events-none">
          Click anywhere to predict
        </div>
      )}
    </div>
  );
};

export default ScatterPlot;
