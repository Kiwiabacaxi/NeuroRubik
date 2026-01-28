/**
 * Neural Network Model Loader
 * Loads weights from JSON and performs inference using TensorFlow.js
 */

import * as tf from '@tensorflow/tfjs';

interface LayerData {
  name: string;
  weights: number[][];
  bias: number[];
  weight_shape: number[];
  bias_shape: number[];
}

interface WeightsData {
  architecture: {
    input_size: number;
    hidden1: number;
    hidden2: number;
    output_size: number;
    total_weights: number;
  };
  layers: LayerData[];
}

export class NeuralSolver {
  private model: tf.Sequential | null = null;
  private isLoaded = false;

  constructor() {}

  async loadWeights(jsonData: WeightsData): Promise<void> {
    // Create the model architecture
    const { hidden1, hidden2 } = jsonData.architecture;
    
    this.model = tf.sequential();
    
    // Input layer + first hidden
    this.model.add(tf.layers.dense({
      units: hidden1,
      activation: 'relu',
      inputShape: [324] // 54 * 6 one-hot
    }));
    
    // Second hidden
    this.model.add(tf.layers.dense({
      units: hidden2,
      activation: 'relu'
    }));
    
    // Output layer
    this.model.add(tf.layers.dense({
      units: 18,
      activation: 'softmax'
    }));

    // Load weights from JSON
    const weights: tf.Tensor[] = [];
    
    for (const layer of jsonData.layers) {
      // Transpose weights from Python (out, in) to TF.js (in, out)
      const weightsArray = layer.weights;
      const transposed = this.transpose2D(weightsArray);
      
      weights.push(tf.tensor2d(transposed));
      weights.push(tf.tensor1d(layer.bias));
    }
    
    this.model.setWeights(weights);
    this.isLoaded = true;
    
    console.log('Model loaded successfully');
    console.log('Architecture:', jsonData.architecture);
  }

  private transpose2D(arr: number[][]): number[][] {
    if (arr.length === 0) return [];
    const rows = arr.length;
    const cols = arr[0].length;
    const result: number[][] = [];
    
    for (let c = 0; c < cols; c++) {
      result[c] = [];
      for (let r = 0; r < rows; r++) {
        result[c][r] = arr[r][c];
      }
    }
    
    return result;
  }

  async loadFromFile(file: File): Promise<void> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = async (event) => {
        try {
          const jsonData = JSON.parse(event.target?.result as string);
          await this.loadWeights(jsonData);
          resolve();
        } catch (error) {
          reject(error);
        }
      };
      
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }

  predictAction(state: Float32Array): number {
    if (!this.model || !this.isLoaded) {
      throw new Error('Model not loaded');
    }

    const inputTensor = tf.tensor2d([Array.from(state)], [1, 324]);
    const output = this.model.predict(inputTensor) as tf.Tensor;
    const probs = output.dataSync();
    
    // Find max probability
    let maxIdx = 0;
    let maxProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIdx = i;
      }
    }
    
    // Cleanup tensors
    inputTensor.dispose();
    output.dispose();
    
    return maxIdx;
  }

  isModelLoaded(): boolean {
    return this.isLoaded;
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isLoaded = false;
    }
  }
}
