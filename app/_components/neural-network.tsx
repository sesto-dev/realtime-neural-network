"use client"

import React, { useState, useEffect, useCallback } from "react"
import { Card } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import * as tf from "@tensorflow/tfjs"
import { DQNAgent } from "./dino-game"

interface Neuron {
  x: number
  y: number
  layer: number
  index: number
  activation: number
}

interface NeuralNetworkProps {
  agent: DQNAgent
  inputValues: number[]
  outputValues: number[]
  updateInputValues: (newInputs: number[]) => void
}

// Update architecture: input is now 4 values.
const NETWORK_ARCHITECTURE = [4, 16, 16, 2]

export default function NeuralNetwork({
  agent,
  inputValues,
  outputValues,
  updateInputValues,
}: NeuralNetworkProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const [neurons, setNeurons] = useState<Neuron[]>([])
  const [weightSummaries, setWeightSummaries] = useState<string[]>([])

  // Calculate positions for neurons in each layer.
  useEffect(() => {
    const newNeurons: Neuron[] = []
    const canvasWidth = 700
    const canvasHeight = 400
    NETWORK_ARCHITECTURE.forEach((size, layerIndex) => {
      for (let i = 0; i < size; i++) {
        newNeurons.push({
          x: ((layerIndex + 1) * canvasWidth) / (NETWORK_ARCHITECTURE.length + 1),
          y: ((i + 1) * canvasHeight) / (size + 1),
          layer: layerIndex,
          index: i,
          activation: 0,
        })
      }
    })
    setNeurons(newNeurons)
  }, [])

  const updateActivations = useCallback(async () => {
    if (!agent || !agent.model) return
    const inputTensor = tf.tensor2d([inputValues])
    const prediction = agent.model.predict(inputTensor) as tf.Tensor
    await prediction.data() // force computation
    const layerOutputs = await Promise.all(
      agent.model.layers.map(async (layer) => {
        const layerModel = tf.model({
          inputs: agent.model.inputs,
          outputs: layer.output as tf.SymbolicTensor,
        })
        const outputTensor = layerModel.predict(inputTensor) as tf.Tensor
        const data = await outputTensor.data()
        outputTensor.dispose()
        return data
      })
    )
    setNeurons((prev) =>
      prev.map((neuron) => {
        if (neuron.layer === 0) {
          return { ...neuron, activation: inputValues[neuron.index] }
        } else {
          const layerData = layerOutputs[neuron.layer - 1]
          if (layerData && neuron.index < layerData.length) {
            return { ...neuron, activation: layerData[neuron.index] }
          }
          return neuron
        }
      })
    )
    const summaries: string[] = []
    agent.model.layers.forEach((layer, idx) => {
      const weights = layer.getWeights()
      if (weights.length > 0) {
        const weightData = weights[0].dataSync()
        let sum = 0
        for (let i = 0; i < weightData.length; i++) {
          sum += Math.abs(weightData[i])
        }
        const avg = sum / weightData.length
        summaries.push(`Layer ${idx + 1} Avg Weight: ${avg.toFixed(4)}`)
      } else {
        summaries.push(`Layer ${idx + 1}: No weights`)
      }
    })
    setWeightSummaries(summaries)
    inputTensor.dispose()
    prediction.dispose()
  }, [agent, inputValues])

  useEffect(() => {
    updateActivations()
  }, [inputValues, updateActivations])

  const drawNetwork = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.strokeStyle = "rgba(148, 163, 184, 0.2)"
    ctx.lineWidth = 1
    neurons.forEach((neuron) => {
      const nextLayerNeurons = neurons.filter((n) => n.layer === neuron.layer + 1)
      nextLayerNeurons.forEach((target) => {
        ctx.beginPath()
        ctx.moveTo(neuron.x, neuron.y)
        ctx.lineTo(target.x, target.y)
        ctx.stroke()
      })
    })
    neurons.forEach((neuron) => {
      ctx.beginPath()
      ctx.arc(neuron.x, neuron.y, 20, 0, Math.PI * 2)
      const intensity = Math.min(Math.abs(neuron.activation) / 1, 1)
      const gradient = ctx.createRadialGradient(neuron.x, neuron.y, 0, neuron.x, neuron.y, 20)
      gradient.addColorStop(0, `rgba(59, 130, 246, ${intensity})`)
      gradient.addColorStop(1, "rgba(59, 130, 246, 0.1)")
      ctx.fillStyle = gradient
      ctx.fill()
      ctx.strokeStyle = "rgba(59, 130, 246, 0.5)"
      ctx.stroke()
      ctx.fillStyle = "#000"
      ctx.font = "10px monospace"
      ctx.textAlign = "center"
      ctx.fillText(neuron.activation.toFixed(2), neuron.x, neuron.y + 30)
    })
  }, [neurons])

  useEffect(() => {
    drawNetwork()
  }, [neurons, drawNetwork])

  return (
    <Card className="p-6 w-full max-w-3xl mx-auto">
      <div className="space-y-6">
        <h3 className="text-2xl font-medium">DQN Network Visualization</h3>
        <div className="grid grid-cols-4 gap-4">
          {["Distance", "Obstacle Speed", "Obstacle Width", "Dino Velocity"].map((label, idx) => (
            <div key={idx}>
              <p className="text-sm font-medium">{label}</p>
              <Slider
                value={[inputValues[idx]]}
                min={0}
                max={1}
                step={0.01}
                onValueChange={([val]) => {
                  const newInputs = [...inputValues]
                  newInputs[idx] = val
                  updateInputValues(newInputs)
                }}
              />
              <p className="text-sm">{inputValues[idx].toFixed(2)}</p>
            </div>
          ))}
        </div>
        <canvas
          ref={canvasRef}
          width={700}
          height={400}
          className="w-full border rounded-lg bg-background"
        />
        <div className="mt-4">
          <p className="text-sm font-medium">Output Q‑Values:</p>
          <p className="text-sm font-mono">
            Do Nothing: {outputValues[0].toFixed(2)} | Jump: {outputValues[1].toFixed(2)}
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
  )
}

