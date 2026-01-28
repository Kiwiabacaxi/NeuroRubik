/**
 * 3D Rubik's Cube Renderer
 * Based on proven approach using materials directly on cube faces
 * and attach() for proper transform preservation during rotations
 */

import * as THREE from 'three';
import { CubeState, Move, MOVES } from '../cube/CubeState';
import { SceneManager } from './SceneManager';

// Configuration
const CUBE_SIZE = 1;
const SPACING = 0.05;
const OFFSET = CUBE_SIZE + SPACING;

// Colors matching the standard cube
const COLORS = {
  R: 0x0045ad,  // Blue (Right)
  L: 0x00a020,  // Green (Left)
  U: 0xffffff,  // White (Up)
  D: 0xffd500,  // Yellow (Down)
  F: 0xe02020,  // Red (Front)
  B: 0xff8000,  // Orange (Back)
  Core: 0x111111
};

// Axes for rotation
const AXES = {
  X: new THREE.Vector3(1, 0, 0),
  Y: new THREE.Vector3(0, 1, 0),
  Z: new THREE.Vector3(0, 0, 1)
};

export class CubeRenderer {
  private sceneManager: SceneManager;
  private cubies: THREE.Mesh[] = [];
  private pivot: THREE.Object3D;
  
  private isAnimating = false;
  private animationQueue: Array<{ move: Move; resolve: () => void }> = [];
  private animationDuration = 300; // ms per move

  constructor(sceneManager: SceneManager) {
    this.sceneManager = sceneManager;
    this.pivot = new THREE.Object3D();
    this.sceneManager.scene.add(this.pivot);
    
    this.createCube();
  }

  private createCube(): void {
    const geometry = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
    
    // Create edges geometry for black outlines
    const edgesGeometry = new THREE.EdgesGeometry(geometry);
    const edgesMaterial = new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 });

    for (let x = -1; x <= 1; x++) {
      for (let y = -1; y <= 1; y++) {
        for (let z = -1; z <= 1; z++) {
          // Create materials for each face based on position
          // Order: +X, -X, +Y, -Y, +Z, -Z (Right, Left, Up, Down, Front, Back)
          const materials = [
            new THREE.MeshStandardMaterial({ color: x === 1 ? COLORS.R : COLORS.Core }),  // Right (+X)
            new THREE.MeshStandardMaterial({ color: x === -1 ? COLORS.L : COLORS.Core }), // Left (-X)
            new THREE.MeshStandardMaterial({ color: y === 1 ? COLORS.U : COLORS.Core }),  // Up (+Y)
            new THREE.MeshStandardMaterial({ color: y === -1 ? COLORS.D : COLORS.Core }), // Down (-Y)
            new THREE.MeshStandardMaterial({ color: z === 1 ? COLORS.F : COLORS.Core }),  // Front (+Z)
            new THREE.MeshStandardMaterial({ color: z === -1 ? COLORS.B : COLORS.Core }), // Back (-Z)
          ];

          const cubie = new THREE.Mesh(geometry, materials);
          cubie.position.set(x * OFFSET, y * OFFSET, z * OFFSET);
          
          // Add black edges
          const edges = new THREE.LineSegments(edgesGeometry, edgesMaterial);
          cubie.add(edges);

          cubie.castShadow = true;
          cubie.receiveShadow = true;

          this.sceneManager.scene.add(cubie);
          this.cubies.push(cubie);
        }
      }
    }
  }

  // Queue a move for animation
  async animateMove(move: Move): Promise<void> {
    return new Promise((resolve) => {
      this.animationQueue.push({ move, resolve });
      this.processQueue();
    });
  }

  private processQueue(): void {
    if (this.isAnimating || this.animationQueue.length === 0) return;
    
    const { move, resolve } = this.animationQueue.shift()!;
    this.executeMove(move, resolve);
  }

  private executeMove(move: Move, onComplete: () => void): void {
    this.isAnimating = true;

    const baseMove = move[0] as 'R' | 'L' | 'U' | 'D' | 'F' | 'B';
    const isPrime = move.includes("'");
    const isDouble = move.includes('2');
    
    const { axisName, layerVal } = this.getMoveParams(baseMove);
    
    // Direction: normal = -1, prime = +1 (reversed for correct rotation)
    let dir = isPrime ? 1 : -1;
    
    // Special cases for L, D, B (inverted axes)
    if (baseMove === 'L' || baseMove === 'D' || baseMove === 'B') {
      dir = -dir;
    }
    
    const targetAngle = (Math.PI / 2) * (isDouble ? 2 : 1) * dir;
    
    this.rotateLayer(axisName, layerVal, targetAngle, onComplete);
  }

  private getMoveParams(baseMove: string): { axisName: 'X' | 'Y' | 'Z'; layerVal: number } {
    switch (baseMove) {
      case 'R': return { axisName: 'X', layerVal: 1 };
      case 'L': return { axisName: 'X', layerVal: -1 };
      case 'U': return { axisName: 'Y', layerVal: 1 };
      case 'D': return { axisName: 'Y', layerVal: -1 };
      case 'F': return { axisName: 'Z', layerVal: 1 };
      case 'B': return { axisName: 'Z', layerVal: -1 };
      default: throw new Error(`Unknown move: ${baseMove}`);
    }
  }

  private rotateLayer(
    axisName: 'X' | 'Y' | 'Z',
    layerVal: number,
    targetAngle: number,
    onComplete: () => void
  ): void {
    // Find cubies in this layer
    const activeCubies: THREE.Mesh[] = [];
    const epsilon = 0.1;
    const worldPosCriteria = layerVal * OFFSET;

    this.cubies.forEach(cubie => {
      cubie.updateMatrixWorld();
      const pos = cubie.position.clone();
      
      let match = false;
      if (axisName === 'X' && Math.abs(pos.x - worldPosCriteria) < epsilon) match = true;
      if (axisName === 'Y' && Math.abs(pos.y - worldPosCriteria) < epsilon) match = true;
      if (axisName === 'Z' && Math.abs(pos.z - worldPosCriteria) < epsilon) match = true;
      
      if (match) activeCubies.push(cubie);
    });

    // Reset pivot
    this.pivot.rotation.set(0, 0, 0);
    this.pivot.position.set(0, 0, 0);

    // Attach cubies to pivot (preserves world transform)
    activeCubies.forEach(cubie => this.pivot.attach(cubie));

    // Animation
    const axisVec = AXES[axisName];
    const startTime = Date.now();

    const animate = () => {
      const now = Date.now();
      let progress = (now - startTime) / this.animationDuration;
      if (progress > 1) progress = 1;
      
      // Ease in-out
      const ease = progress < 0.5 
        ? 2 * progress * progress 
        : 1 - Math.pow(-2 * progress + 2, 2) / 2;

      // Apply rotation
      this.pivot.rotation.set(0, 0, 0);
      this.pivot.rotateOnAxis(axisVec, targetAngle * ease);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        // Final rotation
        this.pivot.rotation.set(0, 0, 0);
        this.pivot.rotateOnAxis(axisVec, targetAngle);
        this.pivot.updateMatrixWorld();

        // Detach cubies back to scene (preserves world transform)
        activeCubies.forEach(cubie => {
          this.sceneManager.scene.attach(cubie);
          // Round positions to avoid floating point drift
          cubie.position.x = Math.round(cubie.position.x * 100) / 100;
          cubie.position.y = Math.round(cubie.position.y * 100) / 100;
          cubie.position.z = Math.round(cubie.position.z * 100) / 100;
        });

        this.isAnimating = false;
        onComplete();
        this.processQueue();
      }
    };

    requestAnimationFrame(animate);
  }

  // Get current cube state from 3D representation
  getCubeState(): CubeState {
    const state = new CubeState();
    
    // For neural network, we need to extract state from 3D
    // This maps the visual cube to the internal state representation
    this.cubies.forEach(cubie => {
      cubie.updateMatrixWorld();
      const pos = cubie.position.clone();
      const ix = Math.round(pos.x / OFFSET);
      const iy = Math.round(pos.y / OFFSET);
      const iz = Math.round(pos.z / OFFSET);

      // Map each visible face to state
      if (iy === 1) this.mapFaceToState(state, 'U', ix, iz, cubie, new THREE.Vector3(0, 1, 0));
      if (iy === -1) this.mapFaceToState(state, 'D', ix, iz, cubie, new THREE.Vector3(0, -1, 0));
      if (iz === 1) this.mapFaceToState(state, 'F', ix, iy, cubie, new THREE.Vector3(0, 0, 1));
      if (iz === -1) this.mapFaceToState(state, 'B', ix, iy, cubie, new THREE.Vector3(0, 0, -1));
      if (ix === 1) this.mapFaceToState(state, 'R', iy, iz, cubie, new THREE.Vector3(1, 0, 0));
      if (ix === -1) this.mapFaceToState(state, 'L', iy, iz, cubie, new THREE.Vector3(-1, 0, 0));
    });

    return state;
  }

  private mapFaceToState(
    state: CubeState, 
    faceName: string, 
    coord1: number, 
    coord2: number, 
    cubie: THREE.Mesh, 
    targetNormal: THREE.Vector3
  ): void {
    // Find which material face is pointing in the target direction
    const localNormals = [
      new THREE.Vector3(1, 0, 0),   // 0: +X (R material)
      new THREE.Vector3(-1, 0, 0),  // 1: -X (L material)
      new THREE.Vector3(0, 1, 0),   // 2: +Y (U material)
      new THREE.Vector3(0, -1, 0),  // 3: -Y (D material)
      new THREE.Vector3(0, 0, 1),   // 4: +Z (F material)
      new THREE.Vector3(0, 0, -1)   // 5: -Z (B material)
    ];

    let bestDot = -1;
    let bestMatIndex = 0;
    const cubieRot = cubie.quaternion;

    for (let i = 0; i < 6; i++) {
      const worldNormal = localNormals[i].clone().applyQuaternion(cubieRot);
      const dot = worldNormal.dot(targetNormal);
      if (dot > bestDot) {
        bestDot = dot;
        bestMatIndex = i;
      }
    }

    // Get color from material
    const materials = cubie.material as THREE.MeshStandardMaterial[];
    const color = materials[bestMatIndex].color.getHex();
    
    // Map color to face index
    const colorToFace: { [key: number]: number } = {
      [COLORS.U]: 0,  // White = UP
      [COLORS.D]: 1,  // Yellow = DOWN
      [COLORS.F]: 2,  // Red = FRONT
      [COLORS.B]: 3,  // Orange = BACK
      [COLORS.R]: 4,  // Blue = RIGHT
      [COLORS.L]: 5,  // Green = LEFT
    };

    const faceValue = colorToFace[color] ?? 0;
    
    // Calculate state index based on face and position
    const faceIndex = { 'U': 0, 'D': 1, 'F': 2, 'B': 3, 'R': 4, 'L': 5 }[faceName]!;
    let stateIdx: number;
    
    switch (faceName) {
      case 'U':
        stateIdx = (1 - coord2) * 3 + (coord1 + 1);
        break;
      case 'D':
        stateIdx = (coord2 + 1) * 3 + (coord1 + 1);
        break;
      case 'F':
        stateIdx = (1 - coord2) * 3 + (coord1 + 1);
        break;
      case 'B':
        stateIdx = (1 - coord2) * 3 + (1 - coord1);
        break;
      case 'R':
        stateIdx = (1 - coord1) * 3 + (1 - coord2);
        break;
      case 'L':
        stateIdx = (1 - coord1) * 3 + (coord2 + 1);
        break;
      default:
        stateIdx = 0;
    }
    
    state.state[faceIndex * 9 + stateIdx] = faceValue;
  }

  // Reset to solved state
  reset(): void {
    // Remove all cubies
    this.cubies.forEach(cubie => {
      this.sceneManager.scene.remove(cubie);
    });
    this.cubies = [];
    
    // Recreate cube
    this.createCube();
  }

  // Check if cube is solved
  isSolved(): boolean {
    const state = this.getCubeState();
    return state.isSolved();
  }

  // Count correct stickers
  countCorrectStickers(): number {
    const state = this.getCubeState();
    return state.countCorrectStickers();
  }

  setAnimationSpeed(ms: number): void {
    this.animationDuration = ms;
  }

  isMoving(): boolean {
    return this.isAnimating || this.animationQueue.length > 0;
  }
}
