/**
 * Rubik's Cube State Representation
 * Mirror of the Python implementation for client-side solving
 */

// Move definitions
export const MOVES = [
  'R', "R'", 'R2',
  'L', "L'", 'L2',
  'U', "U'", 'U2',
  'D', "D'", 'D2',
  'F', "F'", 'F2',
  'B', "B'", 'B2'
] as const;

export type Move = typeof MOVES[number];

// Face indices
export const WHITE = 0, YELLOW = 1, RED = 2, ORANGE = 3, BLUE = 4, GREEN = 5;
export const UP = WHITE, DOWN = YELLOW, FRONT = RED, BACK = ORANGE, RIGHT = BLUE, LEFT = GREEN;

// Color values for rendering
export const COLORS = {
  [WHITE]: 0xffffff,
  [YELLOW]: 0xffd500,
  [RED]: 0xe02020,
  [ORANGE]: 0xff8000,
  [BLUE]: 0x0045ad,
  [GREEN]: 0x00a020
};

export const COLOR_NAMES = ['W', 'Y', 'R', 'O', 'B', 'G'];

export class CubeState {
  state: Int8Array;

  constructor(state?: Int8Array) {
    if (state) {
      this.state = new Int8Array(state);
    } else {
      // Solved cube
      this.state = new Int8Array(54);
      for (let face = 0; face < 6; face++) {
        for (let i = 0; i < 9; i++) {
          this.state[face * 9 + i] = face;
        }
      }
    }
  }

  clone(): CubeState {
    return new CubeState(this.state);
  }

  isSolved(): boolean {
    for (let face = 0; face < 6; face++) {
      const start = face * 9;
      for (let i = 0; i < 9; i++) {
        if (this.state[start + i] !== face) return false;
      }
    }
    return true;
  }

  getFace(faceIdx: number): number[] {
    const start = faceIdx * 9;
    return Array.from(this.state.slice(start, start + 9));
  }

  setFace(faceIdx: number, values: number[]): void {
    const start = faceIdx * 9;
    for (let i = 0; i < 9; i++) {
      this.state[start + i] = values[i];
    }
  }

  private rotateFaceCW(faceIdx: number): void {
    const face = this.getFace(faceIdx);
    // 0 1 2    6 3 0
    // 3 4 5 -> 7 4 1
    // 6 7 8    8 5 2
    const rotated = [face[6], face[3], face[0], face[7], face[4], face[1], face[8], face[5], face[2]];
    this.setFace(faceIdx, rotated);
  }

  applyMove(move: Move): CubeState {
    const baseMove = move[0];
    const isPrime = move.includes("'");
    const isDouble = move.includes('2');

    const times = isDouble ? 2 : (isPrime ? 3 : 1);
    
    for (let t = 0; t < times; t++) {
      switch (baseMove) {
        case 'R': this.moveR(); break;
        case 'L': this.moveL(); break;
        case 'U': this.moveU(); break;
        case 'D': this.moveD(); break;
        case 'F': this.moveF(); break;
        case 'B': this.moveB(); break;
      }
    }

    return this;
  }

  applyMoves(moves: Move[]): CubeState {
    for (const move of moves) {
      this.applyMove(move);
    }
    return this;
  }

  private moveR(): void {
    this.rotateFaceCW(RIGHT);
    
    const temp = [this.state[UP*9+2], this.state[UP*9+5], this.state[UP*9+8]];
    
    // Up <- Front
    this.state[UP*9+2] = this.state[FRONT*9+2];
    this.state[UP*9+5] = this.state[FRONT*9+5];
    this.state[UP*9+8] = this.state[FRONT*9+8];
    
    // Front <- Down
    this.state[FRONT*9+2] = this.state[DOWN*9+2];
    this.state[FRONT*9+5] = this.state[DOWN*9+5];
    this.state[FRONT*9+8] = this.state[DOWN*9+8];
    
    // Down <- Back (reversed)
    this.state[DOWN*9+2] = this.state[BACK*9+6];
    this.state[DOWN*9+5] = this.state[BACK*9+3];
    this.state[DOWN*9+8] = this.state[BACK*9+0];
    
    // Back <- Up (reversed)
    this.state[BACK*9+6] = temp[0];
    this.state[BACK*9+3] = temp[1];
    this.state[BACK*9+0] = temp[2];
  }

  private moveL(): void {
    this.rotateFaceCW(LEFT);
    
    const temp = [this.state[UP*9+0], this.state[UP*9+3], this.state[UP*9+6]];
    
    // Up <- Back (reversed)
    this.state[UP*9+0] = this.state[BACK*9+8];
    this.state[UP*9+3] = this.state[BACK*9+5];
    this.state[UP*9+6] = this.state[BACK*9+2];
    
    // Back <- Down (reversed)
    this.state[BACK*9+8] = this.state[DOWN*9+0];
    this.state[BACK*9+5] = this.state[DOWN*9+3];
    this.state[BACK*9+2] = this.state[DOWN*9+6];
    
    // Down <- Front
    this.state[DOWN*9+0] = this.state[FRONT*9+0];
    this.state[DOWN*9+3] = this.state[FRONT*9+3];
    this.state[DOWN*9+6] = this.state[FRONT*9+6];
    
    // Front <- Up
    this.state[FRONT*9+0] = temp[0];
    this.state[FRONT*9+3] = temp[1];
    this.state[FRONT*9+6] = temp[2];
  }

  private moveU(): void {
    this.rotateFaceCW(UP);
    
    const temp = [this.state[FRONT*9+0], this.state[FRONT*9+1], this.state[FRONT*9+2]];
    
    // Front <- Right
    this.state[FRONT*9+0] = this.state[RIGHT*9+0];
    this.state[FRONT*9+1] = this.state[RIGHT*9+1];
    this.state[FRONT*9+2] = this.state[RIGHT*9+2];
    
    // Right <- Back
    this.state[RIGHT*9+0] = this.state[BACK*9+0];
    this.state[RIGHT*9+1] = this.state[BACK*9+1];
    this.state[RIGHT*9+2] = this.state[BACK*9+2];
    
    // Back <- Left
    this.state[BACK*9+0] = this.state[LEFT*9+0];
    this.state[BACK*9+1] = this.state[LEFT*9+1];
    this.state[BACK*9+2] = this.state[LEFT*9+2];
    
    // Left <- Front
    this.state[LEFT*9+0] = temp[0];
    this.state[LEFT*9+1] = temp[1];
    this.state[LEFT*9+2] = temp[2];
  }

  private moveD(): void {
    this.rotateFaceCW(DOWN);
    
    const temp = [this.state[FRONT*9+6], this.state[FRONT*9+7], this.state[FRONT*9+8]];
    
    // Front <- Left
    this.state[FRONT*9+6] = this.state[LEFT*9+6];
    this.state[FRONT*9+7] = this.state[LEFT*9+7];
    this.state[FRONT*9+8] = this.state[LEFT*9+8];
    
    // Left <- Back
    this.state[LEFT*9+6] = this.state[BACK*9+6];
    this.state[LEFT*9+7] = this.state[BACK*9+7];
    this.state[LEFT*9+8] = this.state[BACK*9+8];
    
    // Back <- Right
    this.state[BACK*9+6] = this.state[RIGHT*9+6];
    this.state[BACK*9+7] = this.state[RIGHT*9+7];
    this.state[BACK*9+8] = this.state[RIGHT*9+8];
    
    // Right <- Front
    this.state[RIGHT*9+6] = temp[0];
    this.state[RIGHT*9+7] = temp[1];
    this.state[RIGHT*9+8] = temp[2];
  }

  private moveF(): void {
    this.rotateFaceCW(FRONT);
    
    const temp = [this.state[UP*9+6], this.state[UP*9+7], this.state[UP*9+8]];
    
    // Up <- Left (rotated)
    this.state[UP*9+6] = this.state[LEFT*9+8];
    this.state[UP*9+7] = this.state[LEFT*9+5];
    this.state[UP*9+8] = this.state[LEFT*9+2];
    
    // Left <- Down
    this.state[LEFT*9+2] = this.state[DOWN*9+0];
    this.state[LEFT*9+5] = this.state[DOWN*9+1];
    this.state[LEFT*9+8] = this.state[DOWN*9+2];
    
    // Down <- Right (rotated)
    this.state[DOWN*9+0] = this.state[RIGHT*9+6];
    this.state[DOWN*9+1] = this.state[RIGHT*9+3];
    this.state[DOWN*9+2] = this.state[RIGHT*9+0];
    
    // Right <- Up
    this.state[RIGHT*9+0] = temp[0];
    this.state[RIGHT*9+3] = temp[1];
    this.state[RIGHT*9+6] = temp[2];
  }

  private moveB(): void {
    this.rotateFaceCW(BACK);
    
    const temp = [this.state[UP*9+0], this.state[UP*9+1], this.state[UP*9+2]];
    
    // Up <- Right (rotated)
    this.state[UP*9+0] = this.state[RIGHT*9+2];
    this.state[UP*9+1] = this.state[RIGHT*9+5];
    this.state[UP*9+2] = this.state[RIGHT*9+8];
    
    // Right <- Down (rotated)
    this.state[RIGHT*9+2] = this.state[DOWN*9+8];
    this.state[RIGHT*9+5] = this.state[DOWN*9+7];
    this.state[RIGHT*9+8] = this.state[DOWN*9+6];
    
    // Down <- Left
    this.state[DOWN*9+6] = this.state[LEFT*9+0];
    this.state[DOWN*9+7] = this.state[LEFT*9+3];
    this.state[DOWN*9+8] = this.state[LEFT*9+6];
    
    // Left <- Up (rotated)
    this.state[LEFT*9+0] = temp[2];
    this.state[LEFT*9+3] = temp[1];
    this.state[LEFT*9+6] = temp[0];
  }

  scramble(numMoves: number = 20): Move[] {
    const moves: Move[] = [];
    let lastBase = '';
    
    for (let i = 0; i < numMoves; i++) {
      let available = MOVES.filter(m => m[0] !== lastBase);
      const move = available[Math.floor(Math.random() * available.length)];
      this.applyMove(move);
      moves.push(move);
      lastBase = move[0];
    }
    
    return moves;
  }

  toOneHot(): Float32Array {
    const oneHot = new Float32Array(54 * 6);
    for (let i = 0; i < 54; i++) {
      oneHot[i * 6 + this.state[i]] = 1.0;
    }
    return oneHot;
  }

  countCorrectStickers(): number {
    let correct = 0;
    for (let face = 0; face < 6; face++) {
      const start = face * 9;
      for (let i = 0; i < 9; i++) {
        if (this.state[start + i] === face) correct++;
      }
    }
    return correct;
  }

  // Get sticker color at a position (for rendering)
  getSticker(face: number, position: number): number {
    return this.state[face * 9 + position];
  }
}
