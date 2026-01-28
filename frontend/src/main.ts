/**
 * Main Application Entry Point
 * Connects all components together
 */

import { SceneManager } from './visualization/SceneManager';
import { CubeRenderer } from './visualization/CubeRenderer';
import { NeuralSolver } from './neural/NeuralSolver';
import { Solver } from './neural/Solver';
import { Move, MOVES, CubeState } from './cube/CubeState';

class App {
  private sceneManager!: SceneManager;
  private cubeRenderer!: CubeRenderer;
  private neuralSolver: NeuralSolver;
  private solver!: Solver;
  
  private moveCount = 0;
  private isSolving = false;

  constructor() {
    this.neuralSolver = new NeuralSolver();
    this.init();
  }

  private init(): void {
    // Get DOM elements
    const container = document.getElementById('canvas-container')!;
    const canvas = document.getElementById('cube-canvas') as HTMLCanvasElement;
    
    // Create scene and renderer
    this.sceneManager = new SceneManager(container, canvas);
    this.cubeRenderer = new CubeRenderer(this.sceneManager);
    this.solver = new Solver(this.neuralSolver);
    
    // Setup UI
    this.setupEventListeners();
    this.updateStats();
    
    console.log('App initialized');
  }

  private setupEventListeners(): void {
    // Load model button
    const btnLoadModel = document.getElementById('btn-load-model')!;
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    
    btnLoadModel.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
    
    // Cube controls
    document.getElementById('btn-scramble')!.addEventListener('click', () => this.scramble());
    document.getElementById('btn-reset')!.addEventListener('click', () => this.reset());
    document.getElementById('btn-solve')!.addEventListener('click', () => this.solve());
    
    // Speed slider
    const speedSlider = document.getElementById('speed-slider') as HTMLInputElement;
    speedSlider.addEventListener('input', () => {
      const speed = parseInt(speedSlider.value);
      this.cubeRenderer.setAnimationSpeed(speed);
    });
    
    // Manual move buttons
    const moveButtons = document.querySelectorAll('.btn-move');
    moveButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const move = btn.getAttribute('data-move') as Move;
        this.applyMove(move);
      });
    });
  }

  private async handleFileUpload(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    
    if (!file) return;
    
    try {
      this.updateStatus('Loading model...');
      await this.neuralSolver.loadFromFile(file);
      
      // Update UI
      const statusEl = document.getElementById('model-status')!;
      statusEl.innerHTML = `
        <span class="status-indicator online"></span>
        <span>Model loaded</span>
      `;
      
      document.getElementById('btn-solve')!.removeAttribute('disabled');
      this.updateStatus('Ready');
      
    } catch (error) {
      console.error('Failed to load model:', error);
      this.updateStatus('Failed to load model');
    }
    
    // Reset file input
    input.value = '';
  }

  private async scramble(): Promise<void> {
    if (this.cubeRenderer.isMoving() || this.isSolving) return;
    
    this.updateStatus('Scrambling...');
    
    // Generate random moves
    const cubeState = new (await import('./cube/CubeState')).CubeState();
    const moves = cubeState.scramble(20);
    
    // Animate each move
    for (const move of moves) {
      await this.cubeRenderer.animateMove(move);
    }
    
    this.moveCount = 0;
    this.updateSolutionDisplay(`Scramble: ${moves.join(' ')}`);
    this.updateStats();
    this.updateStatus('Scrambled');
  }

  private reset(): void {
    if (this.cubeRenderer.isMoving() || this.isSolving) return;
    
    this.cubeRenderer.reset();
    this.moveCount = 0;
    this.updateSolutionDisplay('-');
    this.updateStats();
    this.updateStatus('Reset');
  }

  private async solve(): Promise<void> {
    if (!this.neuralSolver.isModelLoaded() || this.isSolving) return;
    if (this.cubeRenderer.isMoving()) return;
    
    const cubeState = this.cubeRenderer.getCubeState();
    
    if (cubeState.isSolved()) {
      this.updateStatus('Already solved!');
      return;
    }
    
    this.isSolving = true;
    this.updateStatus('Solving...');
    
    const moves: Move[] = [];
    
    try {
      const result = await this.solver.solveAnimated(
        cubeState,
        async (move) => {
          moves.push(move);
          await this.cubeRenderer.animateMove(move);
          this.moveCount++;
          this.updateStats();
          this.updateSolutionDisplay(moves.join(' '));
        },
        (step) => {
          this.updateStatus(`Solving... (${step} moves)`);
        }
      );
      
      if (result.solved) {
        this.updateStatus(`Solved in ${result.moves.length} moves!`);
      } else {
        this.updateStatus(`Stopped (${result.finalCorrect}/54 correct)`);
      }
      
    } catch (error) {
      console.error('Solve error:', error);
      this.updateStatus('Solve failed');
    }
    
    this.isSolving = false;
  }

  private async applyMove(move: Move): Promise<void> {
    if (this.cubeRenderer.isMoving() || this.isSolving) return;
    
    await this.cubeRenderer.animateMove(move);
    this.moveCount++;
    this.updateStats();
  }

  private updateStatus(status: string): void {
    document.getElementById('stat-status')!.textContent = status;
  }

  private updateStats(): void {
    document.getElementById('stat-moves')!.textContent = String(this.moveCount);
    
    const cubeState = this.cubeRenderer.getCubeState();
    const correct = cubeState.countCorrectStickers();
    document.getElementById('stat-correct')!.textContent = `${correct}/54`;
  }

  private updateSolutionDisplay(text: string): void {
    document.getElementById('solution-moves')!.textContent = text;
  }
}

// Start the app
new App();
