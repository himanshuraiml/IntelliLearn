"use client";
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const NeuralNetworkGraph = ({ layers = [2, 4, 4, 1], weights, width = 600, height = 400 }) => {
  const svgRef = useRef();

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 40, right: 40, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Layer X Positions
    const xPositions = layers.map((_, i) => (innerWidth / (layers.length - 1)) * i);

    // Node Data
    const nodes = [];
    layers.forEach((count, lIndex) => {
      const ySpacing = innerHeight / (count + 1);
      for (let i = 0; i < count; i++) {
        nodes.push({
          id: `l${lIndex}n${i}`,
          x: xPositions[lIndex],
          y: ySpacing * (i + 1),
          layer: lIndex,
          index: i
        });
      }
    });

    // Links (Weights)
    const links = [];
    for (let l = 0; l < layers.length - 1; l++) {
      const currentLayerNodes = nodes.filter(n => n.layer === l);
      const nextLayerNodes = nodes.filter(n => n.layer === l + 1);
      
      currentLayerNodes.forEach((source, si) => {
        nextLayerNodes.forEach((target, ti) => {
          let weightValue = 0.5; // Default if weights not provided
          if (weights && weights[l]) {
            weightValue = weights[l][si][ti];
          }

          links.push({
            source,
            target,
            weight: weightValue
          });
        });
      });
    }

    // Draw Links
    g.selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y)
      .attr('stroke', d => d.weight > 0 ? '#0ea5e9' : '#ef4444')
      .attr('stroke-width', d => Math.abs(d.weight) * 2 + 0.5)
      .attr('opacity', 0.4)
      .attr('stroke-dasharray', d => d.weight === 0.5 ? '4' : '0'); // Dash if untyped

    // Draw Nodes
    const nodeGroups = g.selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.x},${d.y})`);

    nodeGroups.append('circle')
      .attr('r', 12)
      .attr('fill', d => d.layer === 0 ? '#64748b' : d.layer === layers.length - 1 ? '#0ea5e9' : '#1e293b')
      .attr('stroke', '#475569')
      .attr('stroke-width', 2);

    nodeGroups.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.3em')
      .attr('fill', 'white')
      .attr('font-size', '8px')
      .text(d => d.layer === 0 ? 'IN' : d.layer === layers.length - 1 ? 'OUT' : 'H');

  }, [layers, weights, width, height]);

  return (
    <div className="w-full h-full">
      <svg ref={svgRef} width={width} height={height} className="rounded-xl" style={{ background: 'rgba(15,23,42,0.6)' }} />
    </div>
  );
};

export default NeuralNetworkGraph;
