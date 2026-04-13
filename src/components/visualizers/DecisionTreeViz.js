"use client";
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const CLASS_COLORS = ['#ef4444', '#0ea5e9', '#10b981', '#f59e0b'];

// Recursively convert internal tree format → d3 hierarchy data
function treeToHierarchy(node, depth = 0) {
  if (!node) return null;
  if (node.leaf) {
    return { name: `Class ${node.label}`, leaf: true, label: node.label, depth };
  }
  const left  = treeToHierarchy(node.left,  depth + 1);
  const right = treeToHierarchy(node.right, depth + 1);
  return {
    name: `${node.feature} ≤ ${node.threshold.toFixed(2)}`,
    feature: node.feature,
    threshold: node.threshold,
    leaf: false,
    depth,
    children: [left, right].filter(Boolean),
  };
}

const DecisionTreeViz = ({ tree, width = 360, height = 260 }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!tree || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 24, right: 16, bottom: 16, left: 16 };
    const iW = width  - margin.left - margin.right;
    const iH = height - margin.top  - margin.bottom;

    const root    = d3.hierarchy(treeToHierarchy(tree));
    const layout  = d3.tree().size([iW, iH]);
    layout(root);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // ── Links ──────────────────────────────────────────────────────────────
    const linkPath = d3.linkVertical().x(d => d.x).y(d => d.y);
    g.selectAll('.dtree-link')
      .data(root.links())
      .join('path')
      .attr('class', 'dtree-link')
      .attr('d', linkPath)
      .attr('fill', 'none')
      .attr('stroke', '#334155')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0)
      .transition().duration(500)
      .attr('opacity', 1);

    // ── Edge labels (yes / no) ─────────────────────────────────────────────
    g.selectAll('.dtree-edge-label')
      .data(root.links())
      .join('text')
      .attr('class', 'dtree-edge-label')
      .attr('x', d => (d.source.x + d.target.x) / 2)
      .attr('y', d => (d.source.y + d.target.y) / 2 - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#4b5563')
      .attr('font-size', 8)
      .text(d => {
        if (!d.source.children) return '';
        return d.source.children[0] === d.target ? 'yes' : 'no';
      });

    // ── Nodes ──────────────────────────────────────────────────────────────
    const nodeGroup = g.selectAll('.dtree-node')
      .data(root.descendants())
      .join('g')
      .attr('class', 'dtree-node')
      .attr('transform', d => `translate(${d.x},${d.y})`)
      .attr('opacity', 0)
      .call(sel => sel.transition().duration(500).attr('opacity', 1));

    // Leaf nodes — colored circle
    nodeGroup.filter(d => d.data.leaf)
      .call(sel => {
        sel.append('circle')
          .attr('r', 13)
          .attr('fill', d => CLASS_COLORS[d.data.label % CLASS_COLORS.length])
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5)
          .attr('opacity', 0.88);
        sel.append('text')
          .attr('dy', '0.35em')
          .attr('text-anchor', 'middle')
          .attr('fill', '#fff')
          .attr('font-size', 9)
          .attr('font-weight', 'bold')
          .text(d => `C${d.data.label}`);
      });

    // Internal nodes — rounded rectangle with split condition
    nodeGroup.filter(d => !d.data.leaf)
      .call(sel => {
        const W = 62, H = 24;
        sel.append('rect')
          .attr('x', -W / 2).attr('y', -H / 2)
          .attr('width', W).attr('height', H)
          .attr('rx', 5)
          .attr('fill', '#1e293b')
          .attr('stroke', '#475569')
          .attr('stroke-width', 1);
        // Feature badge
        sel.append('rect')
          .attr('x', -W / 2 + 2).attr('y', -H / 2 + 3)
          .attr('width', 12).attr('height', H - 6)
          .attr('rx', 3)
          .attr('fill', d => d.data.feature === 'x' ? '#3b82f6' : '#a855f7')
          .attr('opacity', 0.7);
        sel.append('text')
          .attr('x', -W / 2 + 8).attr('dy', '0.35em')
          .attr('text-anchor', 'middle')
          .attr('fill', '#fff')
          .attr('font-size', 7)
          .attr('font-weight', 'bold')
          .text(d => d.data.feature);
        sel.append('text')
          .attr('x', 4).attr('dy', '0.35em')
          .attr('text-anchor', 'middle')
          .attr('fill', '#94a3b8')
          .attr('font-size', 8)
          .attr('font-family', 'monospace')
          .text(d => `≤${d.data.threshold.toFixed(2)}`);
      });

  }, [tree, width, height]);

  return (
    <div className="bg-slate-900/80 rounded-xl border border-white/5 p-3">
      <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-2 flex items-center gap-2">
        Tree Structure
        <span className="text-slate-600 font-normal normal-case text-[10px]">blue=x axis · purple=y axis</span>
      </h3>
      <div className="overflow-x-auto">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          style={{ background: '#0a0f1a', borderRadius: 8, display: 'block' }}
        />
      </div>
    </div>
  );
};

export default DecisionTreeViz;
