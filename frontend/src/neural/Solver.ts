/**
 * Cube Solver using Neural Network
 * Takes a cube state and returns solution moves
 */

import { CubeState, MOVES, Move } from '../cube/CubeState';
import { NeuralSolver } from './NeuralSolver';

export interface SolveResult {
  solved: boolean;
  moves: Move[];
  finalCorrect: number;
}

export class Solver {
  private neuralSolver: NeuralSolver;
  private maxSteps: number;

  constructor(neuralSolver: NeuralSolver, maxSteps: number = 50) {
    this.neuralSolver = neuralSolver;
    this.maxSteps = maxSteps;
  }

  solve(cubeState: CubeState): SolveResult {
    if (!this.neuralSolver.isModelLoaded()) {
      throw new Error('Model not loaded');
    }

    const cube = cubeState.clone();
    const moves: Move[] = [];

    for (let step = 0; step < this.maxSteps; step++) {
      if (cube.isSolved()) {
        return {
          solved: true,
          moves,
          finalCorrect: 54
        };
      }

      // Get prediction from neural network
      const state = cube.toOneHot();
      const actionIdx = this.neuralSolver.predictAction(state);
      const move = MOVES[actionIdx];

      // Apply move
      cube.applyMove(move);
      moves.push(move);

      // Simple loop detection - if we've made the same 4 moves in a row, something's wrong
      if (moves.length >= 4) {
        const last4 = moves.slice(-4);
        if (last4[0] === last4[2] && last4[1] === last4[3]) {
          // Stuck in a loop, break
          break;
        }
      }
    }

    return {
      solved: cube.isSolved(),
      moves,
      finalCorrect: cube.countCorrectStickers()
    };
  }

  async solveAnimated(
    cubeState: CubeState,
    onMove: (move: Move) => Promise<void>,
    onProgress?: (step: number, total: number) => void
  ): Promise<SolveResult> {
    if (!this.neuralSolver.isModelLoaded()) {
      throw new Error('Model not loaded');
    }

    const cube = cubeState.clone();
    const moves: Move[] = [];

    for (let step = 0; step < this.maxSteps; step++) {
      if (cube.isSolved()) {
        return {
          solved: true,
          moves,
          finalCorrect: 54
        };
      }

      if (onProgress) {
        onProgress(step, this.maxSteps);
      }

      // Get prediction from neural network
      const state = cube.toOneHot();
      const actionIdx = this.neuralSolver.predictAction(state);
      const move = MOVES[actionIdx];

      // Apply move
      cube.applyMove(move);
      moves.push(move);

      // Animate the move
      await onMove(move);

      // Loop detection
      if (moves.length >= 4) {
        const last4 = moves.slice(-4);
        if (last4[0] === last4[2] && last4[1] === last4[3]) {
          break;
        }
      }
    }

    return {
      solved: cube.isSolved(),
      moves,
      finalCorrect: cube.countCorrectStickers()
    };
  }
}
