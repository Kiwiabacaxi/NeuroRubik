/**
 * 3D Rubik's Cube Renderer
 * Creates and animates the 3D cube using Three.js
 */

import * as THREE from 'three';
import { CubeState, COLORS, Move, UP, DOWN, FRONT, BACK, RIGHT, LEFT } from '../cube/CubeState';
import { SceneManager } from './SceneManager';

// Cubie positions relative to center
const POSITIONS = [-1, 0, 1];
const CUBIE_SIZE = 0.95;
const GAP = 0.02;

interface Cubie {
  mesh: THREE.Mesh;
  position: THREE.Vector3;
  stickers: THREE.Mesh[];
}

export class CubeRenderer {
  private sceneManager: SceneManager;
  private cubeGroup: THREE.Group;
  private cubies: Cubie[] = [];
  private cubeState: CubeState;
  
  private isAnimating = false;
  private animationQueue: Array<{ move: Move; resolve: () => void }> = [];
  private animationSpeed = 300; // ms per move

  // Animation state
  private currentRotation: {
    axis: THREE.Vector3;
    angle: number;
    targetAngle: number;
    affectedCubies: Cubie[];
    pivotGroup: THREE.Group;
  } | null = null;

  constructor(sceneManager: SceneManager) {
    this.sceneManager = sceneManager;
    this.cubeGroup = new THREE.Group();
    this.cubeState = new CubeState();
    
    this.createCube();
    this.sceneManager.scene.add(this.cubeGroup);
    
    // Register animation callback
    this.sceneManager.onAnimate(this.update.bind(this));
  }

  private createCube(): void {
    // Clear existing cubies
    this.cubies = [];
    this.cubeGroup.clear();

    // Create 26 cubies (27 minus center)
    for (const x of POSITIONS) {
      for (const y of POSITIONS) {
        for (const z of POSITIONS) {
          // Skip center cube
          if (x === 0 && y === 0 && z === 0) continue;
          
          const cubie = this.createCubie(x, y, z);
          this.cubies.push(cubie);
          this.cubeGroup.add(cubie.mesh);
        }
      }
    }

    this.updateColors();
  }

  private createCubie(x: number, y: number, z: number): Cubie {
    // Black base cube
    const geometry = new THREE.BoxGeometry(CUBIE_SIZE, CUBIE_SIZE, CUBIE_SIZE);
    const material = new THREE.MeshStandardMaterial({
      color: 0x111111,
      metalness: 0.3,
      roughness: 0.7
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x * (1 + GAP), y * (1 + GAP), z * (1 + GAP));
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    const stickers: THREE.Mesh[] = [];
    
    // Add sticker faces
    const stickerGeometry = new THREE.PlaneGeometry(0.85, 0.85);
    
    // Right face (x = 1)
    if (x === 1) {
      const sticker = this.createSticker(stickerGeometry);
      sticker.rotation.y = Math.PI / 2;
      sticker.position.x = CUBIE_SIZE / 2 + 0.001;
      mesh.add(sticker);
      stickers.push(sticker);
    }
    
    // Left face (x = -1)
    if (x === -1) {
      const sticker = this.createSticker(stickerGeometry);
      sticker.rotation.y = -Math.PI / 2;
      sticker.position.x = -CUBIE_SIZE / 2 - 0.001;
      mesh.add(sticker);
      stickers.push(sticker);
    }
    
    // Up face (y = 1)
    if (y === 1) {
      const sticker = this.createSticker(stickerGeometry);
      sticker.rotation.x = -Math.PI / 2;
      sticker.position.y = CUBIE_SIZE / 2 + 0.001;
      mesh.add(sticker);
      stickers.push(sticker);
    }
    
    // Down face (y = -1)
    if (y === -1) {
      const sticker = this.createSticker(stickerGeometry);
      sticker.rotation.x = Math.PI / 2;
      sticker.position.y = -CUBIE_SIZE / 2 - 0.001;
      mesh.add(sticker);
      stickers.push(sticker);
    }
    
    // Front face (z = 1)
    if (z === 1) {
      const sticker = this.createSticker(stickerGeometry);
      sticker.position.z = CUBIE_SIZE / 2 + 0.001;
      mesh.add(sticker);
      stickers.push(sticker);
    }
    
    // Back face (z = -1)
    if (z === -1) {
      const sticker = this.createSticker(stickerGeometry);
      sticker.rotation.y = Math.PI;
      sticker.position.z = -CUBIE_SIZE / 2 - 0.001;
      mesh.add(sticker);
      stickers.push(sticker);
    }

    return {
      mesh,
      position: new THREE.Vector3(x, y, z),
      stickers
    };
  }

  private createSticker(geometry: THREE.PlaneGeometry): THREE.Mesh {
    const material = new THREE.MeshStandardMaterial({
      color: 0x333333,
      metalness: 0.1,
      roughness: 0.3,
      side: THREE.FrontSide
    });
    const sticker = new THREE.Mesh(geometry, material);
    return sticker;
  }

  updateColors(): void {
    // Map stickers to cube state
    // This is a simplified version - maps based on cubie position
    
    for (const cubie of this.cubies) {
      const x = Math.round(cubie.position.x);
      const y = Math.round(cubie.position.y);
      const z = Math.round(cubie.position.z);
      
      let stickerIndex = 0;
      
      // Right face
      if (x === 1) {
        const idx = this.posToFaceIndex(RIGHT, y, z);
        const color = this.cubeState.getSticker(RIGHT, idx);
        this.setMaterialColor(cubie.stickers[stickerIndex++], COLORS[color]);
      }
      
      // Left face
      if (x === -1) {
        const idx = this.posToFaceIndex(LEFT, y, z, true);
        const color = this.cubeState.getSticker(LEFT, idx);
        this.setMaterialColor(cubie.stickers[stickerIndex++], COLORS[color]);
      }
      
      // Up face
      if (y === 1) {
        const idx = this.posToUpDownIndex(UP, x, z);
        const color = this.cubeState.getSticker(UP, idx);
        this.setMaterialColor(cubie.stickers[stickerIndex++], COLORS[color]);
      }
      
      // Down face
      if (y === -1) {
        const idx = this.posToUpDownIndex(DOWN, x, z);
        const color = this.cubeState.getSticker(DOWN, idx);
        this.setMaterialColor(cubie.stickers[stickerIndex++], COLORS[color]);
      }
      
      // Front face
      if (z === 1) {
        const idx = this.posToFaceIndex(FRONT, y, x);
        const color = this.cubeState.getSticker(FRONT, idx);
        this.setMaterialColor(cubie.stickers[stickerIndex++], COLORS[color]);
      }
      
      // Back face
      if (z === -1) {
        const idx = this.posToFaceIndex(BACK, y, x, true);
        const color = this.cubeState.getSticker(BACK, idx);
        this.setMaterialColor(cubie.stickers[stickerIndex++], COLORS[color]);
      }
    }
  }

  private posToFaceIndex(_face: number, row: number, col: number, flipCol = false): number {
    // row: -1, 0, 1 -> 0, 1, 2 (inverted because y goes up)
    const r = 1 - row;
    // col: -1, 0, 1 -> 0, 1, 2
    const c = flipCol ? (1 - col) : (col + 1);
    return r * 3 + c;
  }

  private posToUpDownIndex(_face: number, x: number, z: number): number {
    // For UP: looking down
    // x: -1, 0, 1 -> col 0, 1, 2
    // z: 1, 0, -1 -> row 0, 1, 2
    const row = 1 - z;
    const col = x + 1;
    return row * 3 + col;
  }

  private setMaterialColor(mesh: THREE.Mesh, color: number): void {
    (mesh.material as THREE.MeshStandardMaterial).color.setHex(color);
  }

  // Animate a move
  async animateMove(move: Move): Promise<void> {
    return new Promise((resolve) => {
      this.animationQueue.push({ move, resolve });
      this.processQueue();
    });
  }

  private processQueue(): void {
    if (this.isAnimating || this.animationQueue.length === 0) return;
    
    const { move, resolve } = this.animationQueue.shift()!;
    this.startMoveAnimation(move, resolve);
  }

  private startMoveAnimation(move: Move, onComplete: () => void): void {
    this.isAnimating = true;
    
    const baseMove = move[0] as 'R' | 'L' | 'U' | 'D' | 'F' | 'B';
    const isPrime = move.includes("'");
    const isDouble = move.includes('2');
    
    const rotations = isDouble ? 2 : 1;
    const direction = isPrime ? 1 : -1;
    const targetAngle = (Math.PI / 2) * rotations * direction;
    
    // Get axis and affected cubies
    const { axis, affectedCubies } = this.getMoveInfo(baseMove);
    
    // Create pivot group
    const pivotGroup = new THREE.Group();
    this.cubeGroup.add(pivotGroup);
    
    // Move affected cubies to pivot group
    for (const cubie of affectedCubies) {
      const worldPos = new THREE.Vector3();
      cubie.mesh.getWorldPosition(worldPos);
      this.cubeGroup.remove(cubie.mesh);
      pivotGroup.add(cubie.mesh);
    }
    
    this.currentRotation = {
      axis,
      angle: 0,
      targetAngle,
      affectedCubies,
      pivotGroup
    };

    // Actually apply the move to state
    this.cubeState.applyMove(move);

    // Set timeout for animation completion
    setTimeout(() => {
      this.finishMoveAnimation(onComplete);
    }, this.animationSpeed);
  }

  private getMoveInfo(baseMove: string): { axis: THREE.Vector3; affectedCubies: Cubie[] } {
    let axis: THREE.Vector3;
    let filter: (c: Cubie) => boolean;
    
    switch (baseMove) {
      case 'R':
        axis = new THREE.Vector3(-1, 0, 0);
        filter = (c) => Math.round(c.position.x) === 1;
        break;
      case 'L':
        axis = new THREE.Vector3(1, 0, 0);
        filter = (c) => Math.round(c.position.x) === -1;
        break;
      case 'U':
        axis = new THREE.Vector3(0, -1, 0);
        filter = (c) => Math.round(c.position.y) === 1;
        break;
      case 'D':
        axis = new THREE.Vector3(0, 1, 0);
        filter = (c) => Math.round(c.position.y) === -1;
        break;
      case 'F':
        axis = new THREE.Vector3(0, 0, -1);
        filter = (c) => Math.round(c.position.z) === 1;
        break;
      case 'B':
        axis = new THREE.Vector3(0, 0, 1);
        filter = (c) => Math.round(c.position.z) === -1;
        break;
      default:
        throw new Error(`Unknown move: ${baseMove}`);
    }
    
    return {
      axis,
      affectedCubies: this.cubies.filter(filter)
    };
  }

  private finishMoveAnimation(onComplete: () => void): void {
    if (!this.currentRotation) return;
    
    const { affectedCubies, pivotGroup, targetAngle, axis } = this.currentRotation;
    
    // Apply final rotation
    pivotGroup.rotation.setFromAxisAngle(axis, targetAngle);
    
    // Update cubie positions based on rotation
    for (const cubie of affectedCubies) {
      const worldPos = new THREE.Vector3();
      cubie.mesh.getWorldPosition(worldPos);
      
      // Apply rotation to position
      cubie.position.applyAxisAngle(axis, targetAngle);
      
      // Round to nearest integer
      cubie.position.x = Math.round(cubie.position.x);
      cubie.position.y = Math.round(cubie.position.y);
      cubie.position.z = Math.round(cubie.position.z);
      
      // Move back to main group
      pivotGroup.remove(cubie.mesh);
      cubie.mesh.position.set(
        cubie.position.x * (1 + GAP),
        cubie.position.y * (1 + GAP),
        cubie.position.z * (1 + GAP)
      );
      cubie.mesh.rotation.set(0, 0, 0);
      this.cubeGroup.add(cubie.mesh);
    }
    
    // Remove pivot group
    this.cubeGroup.remove(pivotGroup);
    
    // Update colors
    this.updateColors();
    
    this.currentRotation = null;
    this.isAnimating = false;
    
    onComplete();
    
    // Process next in queue
    this.processQueue();
  }

  private update(delta: number): void {
    if (!this.currentRotation) return;
    
    const { axis, targetAngle, pivotGroup } = this.currentRotation;
    const speed = (Math.PI / 2) / (this.animationSpeed / 1000);
    
    // Animate rotation
    this.currentRotation.angle += speed * delta * Math.sign(targetAngle);
    
    // Clamp to target
    if (Math.abs(this.currentRotation.angle) >= Math.abs(targetAngle)) {
      this.currentRotation.angle = targetAngle;
    }
    
    pivotGroup.rotation.setFromAxisAngle(axis, this.currentRotation.angle);
  }

  // Public methods
  setCubeState(state: CubeState): void {
    this.cubeState = state.clone();
    this.resetCubiePositions();
    this.updateColors();
  }

  getCubeState(): CubeState {
    return this.cubeState.clone();
  }

  private resetCubiePositions(): void {
    let index = 0;
    for (const x of POSITIONS) {
      for (const y of POSITIONS) {
        for (const z of POSITIONS) {
          if (x === 0 && y === 0 && z === 0) continue;
          
          const cubie = this.cubies[index++];
          cubie.position.set(x, y, z);
          cubie.mesh.position.set(x * (1 + GAP), y * (1 + GAP), z * (1 + GAP));
          cubie.mesh.rotation.set(0, 0, 0);
        }
      }
    }
  }

  reset(): void {
    this.cubeState = new CubeState();
    this.resetCubiePositions();
    this.updateColors();
  }

  scramble(numMoves: number = 20): Move[] {
    const moves = this.cubeState.scramble(numMoves);
    this.resetCubiePositions();
    this.updateColors();
    return moves;
  }

  setAnimationSpeed(ms: number): void {
    this.animationSpeed = ms;
  }

  isMoving(): boolean {
    return this.isAnimating || this.animationQueue.length > 0;
  }
}
