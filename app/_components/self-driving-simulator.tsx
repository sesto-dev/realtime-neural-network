"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import * as tf from "@tensorflow/tfjs";

// Simulation constants
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const LANES = [200, 400, 600]; // x positions for 3 lanes
const CAR_Y = 500; // fixed y position for our car
const CAR_WIDTH = 40;
const CAR_HEIGHT = 80;
const OBSTACLE_WIDTH = 40;
const OBSTACLE_HEIGHT = 80;
const OBSTACLE_SPEED_BASE = 200; // base speed (pixels per second)
const SPAWN_INTERVAL = 1.5; // seconds between obstacle spawns
const FIXED_DT = 0.02; // fixed simulation time-step (seconds)

// RL parameters
const ACTIONS = ["NOOP", "LEFT", "RIGHT", "ACCEL", "BRAKE"]; // 5 discrete actions

export interface Transition {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

interface PriorityTransition {
  transition: Transition;
  priority: number;
}

// Self‑Driving Environment Class
class SelfDrivingEnvironment {
  carLane: number;
  carSpeed: number; // in pixels per second
  obstacles: { lane: number; y: number }[];
  timeSinceLastSpawn: number;
  distanceDriven: number;
  gameOver: boolean;

  constructor() {
    this.carLane = 1; // start in middle lane
    this.carSpeed = 150; // initial speed
    this.obstacles = [];
    this.timeSinceLastSpawn = 0;
    this.distanceDriven = 0;
    this.gameOver = false;
  }

  reset(): number[] {
    this.carLane = 1;
    this.carSpeed = 150;
    this.obstacles = [];
    this.timeSinceLastSpawn = 0;
    this.distanceDriven = 0;
    this.gameOver = false;
    return this.getSensorState();
  }

  spawnObstacle() {
    const lane = Math.floor(Math.random() * LANES.length);
    this.obstacles.push({ lane, y: -OBSTACLE_HEIGHT });
  }

  updatePhysics(dt: number) {
    // Update distance driven
    this.distanceDriven += this.carSpeed * dt;
    // Obstacles move downward (faster if the car speeds up)
    const obstacleSpeed = OBSTACLE_SPEED_BASE + this.carSpeed;
    this.obstacles.forEach((ob) => {
      ob.y += obstacleSpeed * dt;
    });
    // Remove off‑screen obstacles
    this.obstacles = this.obstacles.filter(
      (ob) => ob.y < CANVAS_HEIGHT + OBSTACLE_HEIGHT
    );
    // Spawn obstacles at regular intervals
    this.timeSinceLastSpawn += dt;
    if (this.timeSinceLastSpawn >= SPAWN_INTERVAL) {
      this.spawnObstacle();
      this.timeSinceLastSpawn = 0;
    }
  }

  checkCollision(): boolean {
    // Check collision if an obstacle in the same lane overlaps the car
    for (let ob of this.obstacles) {
      if (ob.lane === this.carLane) {
        const carX = LANES[this.carLane] - CAR_WIDTH / 2;
        const carY = CAR_Y - CAR_HEIGHT;
        const obX = LANES[ob.lane] - OBSTACLE_WIDTH / 2;
        const obY = ob.y;
        if (
          carX < obX + OBSTACLE_WIDTH &&
          carX + CAR_WIDTH > obX &&
          carY < obY + OBSTACLE_HEIGHT &&
          carY + CAR_HEIGHT > obY
        ) {
          return true;
        }
      }
    }
    return false;
  }

  step(
    action: number,
    dt: number
  ): { nextState: number[]; reward: number; done: boolean } {
    // Actions: 0 = NOOP, 1 = LEFT, 2 = RIGHT, 3 = ACCEL, 4 = BRAKE
    if (action === 1 && this.carLane > 0) {
      this.carLane -= 1;
    } else if (action === 2 && this.carLane < LANES.length - 1) {
      this.carLane += 1;
    }
    if (action === 3) {
      this.carSpeed += 20;
    } else if (action === 4) {
      this.carSpeed = Math.max(50, this.carSpeed - 20);
    }
    // Update simulation physics
    this.updatePhysics(dt);
    if (this.checkCollision()) {
      this.gameOver = true;
      return { nextState: this.getSensorState(), reward: -100, done: true };
    }
    // Reward based on distance traveled in this step (scaled arbitrarily)
    const reward = (dt * this.carSpeed) / 100;
    return { nextState: this.getSensorState(), reward, done: this.gameOver };
  }

  getSensorState(): number[] {
    // Sensor state: [frontDistance, leftDistance, rightDistance, normalizedSpeed]
    let frontDistance = 1;
    let leftDistance = 1;
    let rightDistance = 1;
    for (let ob of this.obstacles) {
      if (ob.lane === this.carLane && ob.y >= 0) {
        const dist = (ob.y - CAR_Y + OBSTACLE_HEIGHT) / CANVAS_HEIGHT;
        if (dist < frontDistance) frontDistance = dist;
      }
      if (ob.lane === this.carLane - 1 && ob.y >= 0) {
        const dist = (ob.y - CAR_Y + OBSTACLE_HEIGHT) / CANVAS_HEIGHT;
        if (dist < leftDistance) leftDistance = dist;
      }
      if (ob.lane === this.carLane + 1 && ob.y >= 0) {
        const dist = (ob.y - CAR_Y + OBSTACLE_HEIGHT) / CANVAS_HEIGHT;
        if (dist < rightDistance) rightDistance = dist;
      }
    }
    // Normalize speed (assume range [50, 300])
    const normSpeed = (this.carSpeed - 50) / (300 - 50);
    return [frontDistance, leftDistance, rightDistance, normSpeed];
  }
}

// DQNAgent with target network and prioritized experience replay
export class DQNAgent {
  model: tf.Sequential;
  targetModel: tf.Sequential;
  replayBuffer: PriorityTransition[] = [];
  epsilon: number = 1.0;
  epsilonDecay: number = 0.995;
  epsilonMin: number = 0.01;
  gamma: number = 0.99;
  batchSize: number = 32;
  targetUpdateFrequency: number = 10; // update target model every 10 episodes

  constructor() {
    this.model = tf.sequential();
    this.model.add(
      tf.layers.dense({ units: 16, inputShape: [4], activation: "relu" })
    );
    this.model.add(tf.layers.dense({ units: 16, activation: "relu" }));
    this.model.add(
      tf.layers.dense({ units: ACTIONS.length, activation: "linear" })
    );
    this.model.compile({
      optimizer: tf.train.adam(0.0005),
      loss: "meanSquaredError",
    });

    this.targetModel = tf.sequential();
    this.targetModel.add(
      tf.layers.dense({ units: 16, inputShape: [4], activation: "relu" })
    );
    this.targetModel.add(tf.layers.dense({ units: 16, activation: "relu" }));
    this.targetModel.add(
      tf.layers.dense({ units: ACTIONS.length, activation: "linear" })
    );
    this.targetModel.compile({
      optimizer: tf.train.adam(0.0005),
      loss: "meanSquaredError",
    });
    this.updateTargetModel();
  }

  updateTargetModel() {
    const weights = this.model.getWeights();
    this.targetModel.setWeights(weights);
  }

  selectAction(state: number[]): number {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * ACTIONS.length);
    }
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state]);
      const qValues = this.model.predict(stateTensor) as tf.Tensor;
      const qArray = qValues.dataSync();
      return qArray.indexOf(Math.max(...qArray));
    });
  }

  storeTransition(transition: Transition) {
    const defaultPriority = 1.0;
    let maxPriority = defaultPriority;
    for (const item of this.replayBuffer) {
      if (item.priority > maxPriority) maxPriority = item.priority;
    }
    this.replayBuffer.push({ transition, priority: maxPriority });
    if (this.replayBuffer.length > 10000) {
      this.replayBuffer.shift();
    }
  }

  async trainOnBatch(): Promise<number | null> {
    if (this.replayBuffer.length < this.batchSize) return null;
    const miniBatch: Array<{ index: number; data: Transition }> = [];
    const priorities = this.replayBuffer.map((item) => item.priority);
    const sumPriorities = priorities.reduce((a, b) => a + b, 0);
    for (let i = 0; i < this.batchSize; i++) {
      let rand = Math.random() * sumPriorities;
      let cumulative = 0;
      for (let j = 0; j < this.replayBuffer.length; j++) {
        cumulative += this.replayBuffer[j].priority;
        if (rand <= cumulative) {
          miniBatch.push({ index: j, data: this.replayBuffer[j].transition });
          break;
        }
      }
    }
    const states = miniBatch.map((item) => item.data.state);
    const nextStates = miniBatch.map((item) => item.data.nextState);
    const statesTensor = tf.tensor2d(states);
    const nextStatesTensor = tf.tensor2d(nextStates);
    const qValues = this.model.predict(statesTensor) as tf.Tensor;
    const qValuesNext = this.targetModel.predict(nextStatesTensor) as tf.Tensor;
    const qValuesArray = qValues.arraySync() as number[][];
    const qValuesNextArray = qValuesNext.arraySync() as number[][];
    const targetQs: number[][] = [];
    const tdErrors: number[] = [];
    for (let i = 0; i < miniBatch.length; i++) {
      const t = miniBatch[i].data;
      let target = t.reward;
      if (!t.done) {
        target += this.gamma * Math.max(...qValuesNextArray[i]);
      }
      const targetVector = [...qValuesArray[i]];
      const tdError = Math.abs(target - targetVector[t.action]);
      tdErrors.push(tdError);
      targetVector[t.action] = target;
      targetQs.push(targetVector);
    }
    const targetTensor = tf.tensor2d(targetQs);
    const history = await this.model.fit(statesTensor, targetTensor, {
      epochs: 1,
      verbose: 0,
    });
    tf.dispose([
      statesTensor,
      nextStatesTensor,
      qValues,
      qValuesNext,
      targetTensor,
    ]);
    const lossValue = history.history.loss
      ? (history.history.loss[0] as number)
      : null;
    for (let i = 0; i < miniBatch.length; i++) {
      const idx = miniBatch[i].index;
      this.replayBuffer[idx].priority = tdErrors[i] + 1e-6;
    }
    return lossValue;
  }
}

// Self‑Driving Simulator Component
interface SelfDrivingSimulatorProps {
  agent: DQNAgent;
  updateNetworkVisualization: (inputs: number[], outputs: number[]) => void;
  updateSimulationStats: (stats: {
    isTraining: boolean;
    episodeCount: number;
    currentScore: number;
    bestScore: number;
    epsilon: number;
    loss: number | null;
  }) => void;
  simSpeed: number;
}

export default function SelfDrivingSimulator({
  agent,
  updateNetworkVisualization,
  updateSimulationStats,
  simSpeed,
}: SelfDrivingSimulatorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [episodeCount, setEpisodeCount] = useState(0);
  const [currentScore, setCurrentScore] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const [epsilon, setEpsilon] = useState(1.0);
  const [loss, setLoss] = useState<number | null>(null);

  const envRef = useRef(new SelfDrivingEnvironment());
  const animationFrameRef = useRef<number>(0);
  const simSpeedRef = useRef(simSpeed);
  useEffect(() => {
    simSpeedRef.current = simSpeed;
  }, [simSpeed]);

  const drawSimulation = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    // Minimalistic dark background
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Draw lane markers
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 2;
    for (let i = 1; i < LANES.length; i++) {
      ctx.beginPath();
      ctx.moveTo(LANES[i] - 100, 0);
      ctx.lineTo(LANES[i] - 100, canvas.height);
      ctx.stroke();
    }
    // Draw obstacles
    ctx.fillStyle = "#e74c3c";
    envRef.current.obstacles.forEach((ob) => {
      const x = LANES[ob.lane] - OBSTACLE_WIDTH / 2;
      const y = ob.y;
      ctx.fillRect(x, y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT);
    });
    // Draw our car
    ctx.fillStyle = "#3498db";
    const carX = LANES[envRef.current.carLane] - CAR_WIDTH / 2;
    const carY = CAR_Y - CAR_HEIGHT;
    ctx.fillRect(carX, carY, CAR_WIDTH, CAR_HEIGHT);
  }, []);

  useEffect(() => {
    const renderLoop = () => {
      drawSimulation();
      animationFrameRef.current = requestAnimationFrame(renderLoop);
    };
    animationFrameRef.current = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animationFrameRef.current);
  }, [drawSimulation]);

  // Simulation loop using fixed dt and multiple iterations per visual frame
  useEffect(() => {
    if (!isRunning) return;
    let running = true;
    async function runEpisode() {
      const env = envRef.current;
      let state = env.reset();
      let done = false;
      let stepCounter = 0;
      while (!done && running) {
        const simulationSteps = Math.max(1, Math.floor(simSpeedRef.current));
        for (let i = 0; i < simulationSteps; i++) {
          stepCounter++;
          const action = agent.selectAction(state);
          const {
            nextState,
            reward,
            done: stepDone,
          } = env.step(action, FIXED_DT);
          agent.storeTransition({
            state,
            action,
            reward,
            nextState,
            done: stepDone,
          });
          state = nextState;
          if (stepCounter % 10 === 0) {
            if ("requestIdleCallback" in window) {
              await new Promise((resolve) =>
                (window as any).requestIdleCallback(resolve)
              );
            }
            const lossResult = await agent.trainOnBatch();
            if (lossResult !== null) {
              setLoss(lossResult);
            }
          }
          if (stepDone) {
            done = true;
            break;
          }
        }
        tf.tidy(() => {
          const inputTensor = tf.tensor2d([state]);
          const prediction = agent.model.predict(inputTensor) as tf.Tensor;
          const output = prediction.dataSync();
          updateNetworkVisualization(state, Array.from(output));
        });
        updateSimulationStats({
          isTraining: isRunning,
          episodeCount: episodeCount + 1,
          currentScore: env.distanceDriven,
          bestScore: bestScore,
          epsilon: agent.epsilon,
          loss: loss,
        });
        await new Promise((res) => setTimeout(res, 20));
      }
      const episodeScore = env.distanceDriven;
      const newEpisodeCount = episodeCount + 1;
      const newBestScore = Math.max(bestScore, episodeScore);
      setCurrentScore(episodeScore);
      setEpisodeCount(newEpisodeCount);
      setBestScore(newBestScore);
      agent.epsilon = Math.max(
        agent.epsilon * agent.epsilonDecay,
        agent.epsilonMin
      );
      setEpsilon(agent.epsilon);
      if (newEpisodeCount % agent.targetUpdateFrequency === 0) {
        agent.updateTargetModel();
      }
    }
    async function episodeLoop() {
      while (running) {
        await runEpisode();
      }
    }
    episodeLoop();
    return () => {
      running = false;
    };
  }, [isRunning]);

  const handleStartPause = () => {
    setIsRunning((prev) => !prev);
  };

  const handleReset = () => {
    setIsRunning(false);
    envRef.current = new SelfDrivingEnvironment();
    agent.replayBuffer = [];
    agent.epsilon = 1.0;
    agent.model.dispose();
    agent.targetModel.dispose();
    const newAgent = new DQNAgent();
    agent.model = newAgent.model;
    agent.targetModel = newAgent.targetModel;
  };

  return (
    <Card className="p-6 w-full max-w-4xl mx-auto">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h3 className="text-2xl font-medium">Self‑Driving Car – RL Agent</h3>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" onClick={handleStartPause}>
              {isRunning ? "Pause" : "Start"}
            </Button>
            <Button variant="outline" size="icon" onClick={handleReset}>
              Reset
            </Button>
          </div>
        </div>
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          className="w-full border rounded-lg"
        />
      </div>
    </Card>
  );
}
