"use client";

import { useState, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import DinoGame, { DQNAgent } from "./dino-game";
import NeuralNetwork from "./neural-network";

export default function Display() {
  // Create a single DQNAgent instance shared by game and visualization.
  const agentRef = useRef(new DQNAgent());

  // Sensor state: [distance, obstacleSpeed, obstacleWidth, dinoVerticalVelocity]
  const [networkInputs, setNetworkInputs] = useState<number[]>([1, 0, 0, 0.5]);
  const [networkOutputs, setNetworkOutputs] = useState<number[]>([0, 0]);
  const [isTraining, setIsTraining] = useState(false);
  const [episodeCount, setEpisodeCount] = useState(0);
  const [currentScore, setCurrentScore] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const [epsilon, setEpsilon] = useState(1.0);
  const [loss, setLoss] = useState<number | null>(null);
  const [gameSpeed, setGameSpeed] = useState(1.0);

  // Called by DinoGame to update the visualization inputs/outputs.
  const updateNetworkVisualization = (inputs: number[], outputs: number[]) => {
    setNetworkInputs(inputs);
    setNetworkOutputs(outputs);
  };

  // Called by DinoGame to update stats.
  const updateGameStats = (stats: {
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

  // Called by NeuralNetwork when the manual slider changes.
  const handleInputChange = (newInputs: number[]) => {
    setNetworkInputs(newInputs);
  };

  return (
    <Card className="p-6 w-full max-w-7xl mx-auto">
      <h2 className="text-3xl font-bold mb-6">
        Dino Game with Neural Network Visualization
      </h2>
      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="text-xl font-semibold mb-4">Game</h3>
          <DinoGame
            agent={agentRef.current}
            updateNetworkVisualization={updateNetworkVisualization}
            updateGameStats={updateGameStats}
            gameSpeed={gameSpeed}
          />
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
          <p className="text-sm font-medium">Score (Cleared Obstacles)</p>
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
        <p className="text-sm font-medium">Game Speed</p>
        <Slider
          value={[gameSpeed]}
          min={1}
          max={5}
          step={0.1}
          onValueChange={([val]) => setGameSpeed(val)}
        />
        <p className="text-sm">{gameSpeed.toFixed(1)}×</p>
      </div>
      <div className="mt-6">
        <p className="text-sm font-medium">Loss</p>
        <p className="text-2xl font-bold">
          {loss !== null ? loss.toFixed(4) : "N/A"}
        </p>
      </div>
    </Card>
  );
}
