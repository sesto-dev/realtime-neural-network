"use client";

import { useState, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import SelfDrivingSimulator, { DQNAgent } from "./self-driving-simulator";
import NeuralNetwork from "./neural-network";

export default function Display() {
  // Create a single DQNAgent instance shared by simulation and visualization.
  const agentRef = useRef(new DQNAgent());

  // Sensor state: [frontDistance, leftDistance, rightDistance, speed]
  const [networkInputs, setNetworkInputs] = useState<number[]>([1, 1, 1, 0.5]);
  // Q-values for actions: [NOOP, LEFT, RIGHT, ACCEL, BRAKE]
  const [networkOutputs, setNetworkOutputs] = useState<number[]>([
    0, 0, 0, 0, 0,
  ]);
  const [isTraining, setIsTraining] = useState(false);
  const [episodeCount, setEpisodeCount] = useState(0);
  const [currentScore, setCurrentScore] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const [epsilon, setEpsilon] = useState(1.0);
  const [loss, setLoss] = useState<number | null>(null);
  const [simSpeed, setSimSpeed] = useState(1.0);

  // Called by SelfDrivingSimulator to update the neural network visualization.
  const updateNetworkVisualization = (inputs: number[], outputs: number[]) => {
    setNetworkInputs(inputs);
    setNetworkOutputs(outputs);
  };

  // Called by SelfDrivingSimulator to update simulation stats.
  const updateSimulationStats = (stats: {
    isTraining: boolean;
    episodeCount: number;
    currentScore: number;
    bestScore: number;
    epsilon: number;
    loss: number | null;
  }) => {
    setIsTraining(stats.isTraining);
    setEpisodeCount(stats.episodeCount);
    setCurrentScore(stats.currentScore);
    setBestScore(stats.bestScore);
    setEpsilon(stats.epsilon);
    setLoss(stats.loss);
  };

  // Allow manual tweaking of inputs via the NN visualization.
  const handleInputChange = (newInputs: number[]) => {
    setNetworkInputs(newInputs);
  };

  return (
    <Card className="p-6 w-full max-w-7xl mx-auto">
      <div className="grid grid-cols-2 gap-6">
        <div className="space-y-4">
          <h3 className="text-xl font-semibold mb-4">
            Neural Network Self‑Driving Car Simulator
          </h3>
          <SelfDrivingSimulator
            agent={agentRef.current}
            updateNetworkVisualization={updateNetworkVisualization}
            updateSimulationStats={updateSimulationStats}
            simSpeed={simSpeed}
          />
          <Card className="p-6">
            <div className="mt-6 grid grid-cols-5 gap-4">
              <div>
                <p className="text-sm font-medium">Status</p>
                <p className="text-2xl font-bold">
                  {isTraining ? "Training" : "Idle"}
                </p>
              </div>
              <div>
                <p className="text-sm font-medium">Episode</p>
                <p className="text-2xl font-bold">{episodeCount}</p>
              </div>
              <div>
                <p className="text-sm font-medium">Score (Distance)</p>
                <p className="text-2xl font-bold">{currentScore}</p>
              </div>
              <div>
                <p className="text-sm font-medium">Best Score</p>
                <p className="text-2xl font-bold">{bestScore}</p>
              </div>
              <div>
                <p className="text-sm font-medium">Epsilon</p>
                <p className="text-2xl font-bold">{epsilon.toFixed(2)}</p>
              </div>
            </div>
            <div className="mt-6">
              <p className="text-sm font-medium">Simulation Speed</p>
              <Slider
                value={[simSpeed]}
                min={1}
                max={20}
                step={0.1}
                onValueChange={([val]) => setSimSpeed(val)}
              />
              <p className="text-sm">{simSpeed.toFixed(1)}×</p>
            </div>
            <div className="mt-6">
              <p className="text-sm font-medium">Training Loss</p>
              <p className="text-2xl font-bold">
                {loss !== null ? loss.toFixed(4) : "N/A"}
              </p>
            </div>
          </Card>
        </div>
        <div>
          <h3 className="text-xl font-semibold mb-4">Neural Network</h3>
          <NeuralNetwork
            agent={agentRef.current}
            inputValues={networkInputs}
            outputValues={networkOutputs}
            updateInputValues={handleInputChange}
          />
        </div>
      </div>
    </Card>
  );
}
