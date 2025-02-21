"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import * as tf from "@tensorflow/tfjs";

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

const GAME_WIDTH = 800;
const GAME_HEIGHT = 250;
const GROUND_Y = GAME_HEIGHT - 20;
const GRAVITY = 0.6;
const JUMP_FORCE = -10;
const BASE_OBSTACLE_SPEED = -4;
const SPAWN_RATE = 0.02;
const MIN_GAP = 200; // Minimum gap (in pixels) between obstacles
const DEFAULT_TARGET_UPDATE_FREQUENCY = 5; // Update target network every 5 episodes
const JUMP_PENALTY_THRESHOLD = 0.8; // Normalized distance threshold for unnecessary jumps
const JUMP_PENALTY = -0.1; // Small penalty for jumping when no obstacle is near

// -----------------------------
// Environment: encapsulates game physics
// -----------------------------
class DinoEnvironment {
  dino: { x: number; y: number; width: number; height: number };
  obstacles: {
    x: number;
    y: number;
    width: number;
    height: number;
    speed: number;
    scored?: boolean;
  }[];
  score: number;
  gameOver: boolean;
  isJumping: boolean;
  jumpVelocity: number;

  constructor() {
    this.dino = { x: 50, y: GROUND_Y, width: 20, height: 40 };
    this.obstacles = [];
    this.score = 0;
    this.gameOver = false;
    this.isJumping = false;
    this.jumpVelocity = 0;
  }

  reset(): number[] {
    this.dino = { x: 50, y: GROUND_Y, width: 20, height: 40 };
    this.obstacles = [];
    this.score = 0;
    this.gameOver = false;
    this.isJumping = false;
    this.jumpVelocity = 0;
    return this.getSensorState();
  }

  updatePhysics(dt: number) {
    // dt is the simulation delta time (in seconds)
    if (this.isJumping) {
      this.jumpVelocity += GRAVITY * dt;
      const newY = this.dino.y + this.jumpVelocity * dt;
      if (newY >= GROUND_Y) {
        this.dino.y = GROUND_Y;
        this.isJumping = false;
        this.jumpVelocity = 0;
      } else {
        this.dino.y = newY;
      }
    }
    this.obstacles.forEach((ob) => {
      ob.x += ob.speed * dt;
      if (!ob.scored && ob.x + ob.width < this.dino.x) {
        this.score += 1;
        ob.scored = true;
      }
    });
    this.obstacles = this.obstacles.filter((ob) => ob.x + ob.width > 0);
  }

  spawnObstacle(dt: number) {
    // Adjust spawn probability by dt to keep consistent obstacle frequency.
    if (
      this.obstacles.length === 0 ||
      Math.max(...this.obstacles.map((ob) => ob.x)) < GAME_WIDTH - MIN_GAP
    ) {
      if (Math.random() < SPAWN_RATE * dt) {
        const width = 15 + Math.random() * 10;
        const height = 15 + Math.random() * 10;
        this.obstacles.push({
          x: GAME_WIDTH,
          y: GROUND_Y,
          width,
          height,
          speed: BASE_OBSTACLE_SPEED,
          scored: false,
        });
      }
    }
  }

  step(
    action: number,
    dt: number
  ): { nextState: number[]; reward: number; done: boolean } {
    // Capture sensor state before action for penalty checking.
    const sensorBefore = this.getSensorState();

    // action: 0 = do nothing, 1 = jump
    if (action === 1 && !this.isJumping && this.dino.y === GROUND_Y) {
      this.isJumping = true;
      this.jumpVelocity = JUMP_FORCE;
    }
    // If jumping when no obstacle is near, apply a small penalty.
    let penalty = 0;
    if (action === 1 && sensorBefore[0] > JUMP_PENALTY_THRESHOLD) {
      penalty = JUMP_PENALTY;
    }
    const prevScore = this.score;
    this.updatePhysics(dt);
    this.spawnObstacle(dt);
    if (this.checkCollision()) {
      this.gameOver = true;
      return { nextState: this.getSensorState(), reward: -100, done: true };
    }
    // Add a small alive reward.
    const aliveReward = 0.01;
    const reward = this.score - prevScore + penalty + aliveReward;
    return { nextState: this.getSensorState(), reward, done: this.gameOver };
  }

  checkCollision(): boolean {
    // Standard AABB collision detection
    for (let ob of this.obstacles) {
      const dinoLeft = this.dino.x;
      const dinoRight = this.dino.x + this.dino.width;
      const dinoTop = this.dino.y - this.dino.height;
      const dinoBottom = this.dino.y;
      const obLeft = ob.x;
      const obRight = ob.x + ob.width;
      const obTop = ob.y - ob.height;
      const obBottom = ob.y;
      if (
        dinoRight > obLeft &&
        dinoLeft < obRight &&
        dinoBottom > obTop &&
        dinoTop < obBottom
      ) {
        return true;
      }
    }
    return false;
  }

  getSensorState(): number[] {
    // Provide richer sensor data:
    // [normalized distance to next obstacle, normalized obstacle speed, normalized obstacle width, normalized dino vertical velocity]
    let distance = 1,
      speed = 0,
      width = 0;
    if (this.obstacles.length > 0) {
      const ob = this.obstacles[0];
      distance = (ob.x - this.dino.x) / GAME_WIDTH;
      if (distance < 0) distance = 0;
      speed = Math.abs(ob.speed / BASE_OBSTACLE_SPEED); // will be 1 if speed is BASE_OBSTACLE_SPEED
      width = ob.width / 50; // assuming maximum width ~50 pixels
    }
    // Normalize dino vertical velocity: assume range [-10, 10]
    const normalizedVelocity = (this.jumpVelocity + 10) / 20;
    return [distance, speed, width, normalizedVelocity];
  }
}

// -----------------------------
// DQN Agent with Target Network and Prioritized Experience Replay
// -----------------------------
export class DQNAgent {
  model: tf.Sequential;
  targetModel: tf.Sequential;
  replayBuffer: PriorityTransition[] = [];
  epsilon: number = 1.0;
  epsilonDecay: number = 0.995;
  epsilonMin: number = 0.01;
  gamma: number = 0.99;
  batchSize: number = 32;
  targetUpdateFrequency: number = DEFAULT_TARGET_UPDATE_FREQUENCY;

  constructor() {
    // Input shape is now [4]
    this.model = tf.sequential();
    this.model.add(
      tf.layers.dense({ units: 16, inputShape: [4], activation: "relu" })
    );
    this.model.add(tf.layers.dense({ units: 16, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 2, activation: "linear" }));
    // Use a lower learning rate.
    this.model.compile({
      optimizer: tf.train.adam(0.0005),
      loss: "meanSquaredError",
    });

    this.targetModel = tf.sequential();
    this.targetModel.add(
      tf.layers.dense({ units: 16, inputShape: [4], activation: "relu" })
    );
    this.targetModel.add(tf.layers.dense({ units: 16, activation: "relu" }));
    this.targetModel.add(tf.layers.dense({ units: 2, activation: "linear" }));
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
      return Math.floor(Math.random() * 2); // Uniform random between 0 and 1
    }
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state]);
      const qValues = this.model.predict(stateTensor) as tf.Tensor;
      const qArray = qValues.dataSync();
      return qArray[0] > qArray[1] ? 0 : 1;
    });
  }

  storeTransition(transition: Transition) {
    const defaultPriority = 1.0;
    let maxPriority = defaultPriority;
    for (const item of this.replayBuffer) {
      if (item.priority > maxPriority) {
        maxPriority = item.priority;
      }
    }
    this.replayBuffer.push({ transition, priority: maxPriority });
    if (this.replayBuffer.length > 10000) {
      this.replayBuffer.shift();
    }
  }

  async trainOnBatch(): Promise<number | null> {
    if (this.replayBuffer.length < this.batchSize) return null;
    // Prioritized sampling
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
    const loss = history.history.loss
      ? (history.history.loss[0] as number)
      : null;
    // Update priorities for sampled transitions.
    for (let i = 0; i < miniBatch.length; i++) {
      const idx = miniBatch[i].index;
      this.replayBuffer[idx].priority = tdErrors[i] + 1e-6;
    }
    return loss;
  }
}

// -----------------------------
// Main Component: DinoGame
// -----------------------------
interface DinoGameProps {
  agent: DQNAgent;
  updateNetworkVisualization: (inputs: number[], outputs: number[]) => void;
  updateGameStats: (stats: {
    isTraining: boolean;
    episodeCount: number;
    currentScore: number;
    bestScore: number;
    epsilon: number;
    loss: number | null;
  }) => void;
  gameSpeed: number;
}

export default function DinoGame({
  agent,
  updateNetworkVisualization,
  updateGameStats,
  gameSpeed,
}: DinoGameProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [episodeCount, setEpisodeCount] = useState(0);
  const [currentScore, setCurrentScore] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const [epsilon, setEpsilon] = useState(1.0);
  const [loss, setLoss] = useState<number | null>(null);

  const envRef = useRef(new DinoEnvironment());
  const animationFrameRef = useRef<number>(0);
  const gameSpeedRef = useRef(gameSpeed);
  useEffect(() => {
    gameSpeedRef.current = gameSpeed;
  }, [gameSpeed]);

  const drawGame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const env = envRef.current;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, GROUND_Y + env.dino.height, canvas.width, 2);
    ctx.fillStyle = "#333";
    ctx.fillRect(
      env.dino.x,
      env.dino.y - env.dino.height,
      env.dino.width,
      env.dino.height
    );
    ctx.fillStyle = "#666";
    env.obstacles.forEach((ob) => {
      ctx.fillRect(ob.x, ob.y - ob.height, ob.width, ob.height);
    });
  }, []);

  useEffect(() => {
    const renderLoop = () => {
      drawGame();
      animationFrameRef.current = requestAnimationFrame(renderLoop);
    };
    animationFrameRef.current = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animationFrameRef.current);
  }, [drawGame]);

  // Episode loop with integrated training updates.
  useEffect(() => {
    if (!isRunning) return;
    let running = true;
    async function runEpisode() {
      const env = envRef.current;
      let state = env.reset();
      let done = false;
      let stepCounter = 0;
      // Use a fixed simulation time-step for physics updates.
      const fixedDt = 1;
      while (!done && running) {
        // Determine the number of simulation iterations per visual frame.
        const simulationSteps = Math.max(1, Math.floor(gameSpeedRef.current));
        for (let i = 0; i < simulationSteps; i++) {
          stepCounter++;
          const action = agent.selectAction(state);
          // Pass the fixed dt so that physics remain consistent.
          const {
            nextState,
            reward,
            done: stepDone,
          } = env.step(action, fixedDt);
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
        // Update visualization once per visual frame.
        tf.tidy(() => {
          const inputTensor = tf.tensor2d([state]);
          const prediction = agent.model.predict(inputTensor) as tf.Tensor;
          const output = prediction.dataSync();
          updateNetworkVisualization(state, Array.from(output));
        });
        updateGameStats({
          isTraining: isRunning,
          episodeCount: episodeCount + 1, // Will update after episode completion.
          currentScore: env.score,
          bestScore: bestScore,
          epsilon: agent.epsilon,
          loss: loss,
        });
        // Use a constant delay for visualization.
        await new Promise((res) => setTimeout(res, 20));
      }
      // After episode completes, update cumulative metrics.
      const episodeScore = env.score;
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
    // Do not reset episodeCount and bestScore so they persist across rounds.
    setCurrentScore(0);
    setEpsilon(1.0);
    setLoss(null);
    envRef.current = new DinoEnvironment();
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
          <h3 className="text-2xl font-medium">Dino Game – RL Agent</h3>
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
          width={GAME_WIDTH}
          height={GAME_HEIGHT}
          className="w-full border rounded-lg bg-background"
        />
      </div>
    </Card>
  );
}
