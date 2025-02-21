"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import * as tf from "@tensorflow/tfjs";
import { DQNAgent } from "./self-driving-simulator";

interface Neuron {
  x: number;
  y: number;
  layer: number;
  index: number;
  activation: number;
}

interface NeuralNetworkProps {
  agent: DQNAgent;
  inputValues: number[];
  outputValues: number[];
  updateInputValues: (newInputs: number[]) => void;
}

// Network architecture: [4, 16, 16, 5]
const NETWORK_ARCHITECTURE = [4, 16, 16, 5];

export default function NeuralNetwork({
  agent,
  inputValues,
  outputValues,
  updateInputValues,
}: NeuralNetworkProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [neurons, setNeurons] = useState<Neuron[]>([]);
  const [weightSummaries, setWeightSummaries] = useState<string[]>([]);
  const activationModels = useRef<Record<number, tf.LayersModel>>({});

  useEffect(() => {
    if (!agent || !agent.model) return;
    const models: Record<number, tf.LayersModel> = {};
    for (let layerIdx = 1; layerIdx < NETWORK_ARCHITECTURE.length; layerIdx++) {
      models[layerIdx] = tf.model({
        inputs: agent.model.inputs,
        outputs: agent.model.layers[layerIdx - 1].output as tf.SymbolicTensor,
      });
    }
    activationModels.current = models;
  }, [agent]);

  useEffect(() => {
    const newNeurons: Neuron[] = [];
    const canvasWidth = 700;
    const canvasHeight = 400;
    NETWORK_ARCHITECTURE.forEach((size, layerIndex) => {
      for (let i = 0; i < size; i++) {
        newNeurons.push({
          x:
            ((layerIndex + 1) * canvasWidth) /
            (NETWORK_ARCHITECTURE.length + 1),
          y: ((i + 1) * canvasHeight) / (size + 1),
          layer: layerIndex,
          index: i,
          activation: 0,
        });
      }
    });
    setNeurons(newNeurons);
  }, []);

  const updateActivations = useCallback(async () => {
    if (!agent || !agent.model) return;
    const activations: Record<number, number[]> = {};
    activations[0] = inputValues;
    const promises = Object.entries(activationModels.current).map(
      async ([layerStr, model]) => {
        const layer = parseInt(layerStr, 10);
        const outputTensor = model.predict(
          tf.tensor2d([inputValues])
        ) as tf.Tensor;
        const data = await outputTensor.data();
        outputTensor.dispose();
        activations[layer] = Array.from(data);
      }
    );
    await Promise.all(promises);
    setNeurons((prev) =>
      prev.map((neuron) => {
        const act = activations[neuron.layer];
        if (act && neuron.index < act.length) {
          return { ...neuron, activation: act[neuron.index] };
        }
        return neuron;
      })
    );
    const summaries: string[] = [];
    agent.model.layers.forEach((layer, idx) => {
      const weights = layer.getWeights();
      if (weights.length > 0) {
        const weightData = weights[0].dataSync();
        let sum = 0;
        for (let i = 0; i < weightData.length; i++) {
          sum += Math.abs(weightData[i]);
        }
        const avg = sum / weightData.length;
        summaries.push(`Layer ${idx + 1} Avg Weight: ${avg.toFixed(4)}`);
      } else {
        summaries.push(`Layer ${idx + 1}: No weights`);
      }
    });
    setWeightSummaries(summaries);
  }, [agent, inputValues]);

  useEffect(() => {
    updateActivations();
  }, [inputValues, updateActivations]);

  const drawNetwork = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
    ctx.lineWidth = 1;
    neurons.forEach((neuron) => {
      const nextLayerNeurons = neurons.filter(
        (n) => n.layer === neuron.layer + 1
      );
      nextLayerNeurons.forEach((target) => {
        ctx.beginPath();
        ctx.moveTo(neuron.x, neuron.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();
      });
    });
    neurons.forEach((neuron) => {
      ctx.beginPath();
      ctx.arc(neuron.x, neuron.y, 20, 0, Math.PI * 2);
      const intensity = Math.min(Math.abs(neuron.activation) / 1, 1);
      const gradient = ctx.createRadialGradient(
        neuron.x,
        neuron.y,
        0,
        neuron.x,
        neuron.y,
        20
      );
      gradient.addColorStop(0, `rgba(52, 152, 219, ${intensity})`);
      gradient.addColorStop(1, "rgba(52, 152, 219, 0.1)");
      ctx.fillStyle = gradient;
      ctx.fill();
      ctx.strokeStyle = "rgba(59, 130, 246, 0.5)";
      ctx.stroke();
      ctx.fillStyle = "#101010";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(neuron.activation.toFixed(2), neuron.x, neuron.y + 30);
    });
  }, [neurons]);

  useEffect(() => {
    drawNetwork();
  }, [neurons, drawNetwork]);

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <h3 className="text-2xl font-medium">DQN Network Visualization</h3>
        <div className="grid grid-cols-4 gap-4">
          {["Front Distance", "Left Distance", "Right Distance", "Speed"].map(
            (label, idx) => (
              <div key={idx}>
                <p className="text-sm font-medium">{label}</p>
                <Slider
                  value={[inputValues[idx]]}
                  min={0}
                  max={1}
                  step={0.01}
                  onValueChange={([val]) => {
                    const newInputs = [...inputValues];
                    newInputs[idx] = val;
                    updateInputValues(newInputs);
                  }}
                />
                <p className="text-sm">{inputValues[idx].toFixed(2)}</p>
              </div>
            )
          )}
        </div>
        <canvas
          ref={canvasRef}
          width={700}
          height={400}
          className="w-full border rounded-lg"
        />
        <div className="mt-4">
          <p className="text-sm font-medium">Output Q‑Values:</p>
          <p className="text-sm font-mono">
            {`NOOP: ${outputValues[0].toFixed(
              2
            )} | LEFT: ${outputValues[1].toFixed(
              2
            )} | RIGHT: ${outputValues[2].toFixed(
              2
            )} | ACCEL: ${outputValues[3].toFixed(
              2
            )} | BRAKE: ${outputValues[4].toFixed(2)}`}
          </p>
        </div>
        <div className="mt-4">
          <p className="text-sm font-medium">Weight Summaries:</p>
          {weightSummaries.map((summary, idx) => (
            <p key={idx} className="text-sm font-mono">
              {summary}
            </p>
          ))}
        </div>
      </div>
    </Card>
  );
}
