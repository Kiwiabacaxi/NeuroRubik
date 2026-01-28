"""
Rubik's Cube State Representation and Movement Logic

The cube is represented as a 54-element array (6 faces × 9 stickers).
Each face is indexed as follows:

Face indices:
    0 = White (Up)
    1 = Yellow (Down)
    2 = Red (Front)
    3 = Orange (Back)
    4 = Blue (Right)
    5 = Green (Left)

Sticker positions on each face (looking at the face):
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

import random
from typing import List, Optional

import numpy as np

# Move names and their indices
MOVES = [
    "R",
    "R'",
    "R2",
    "L",
    "L'",
    "L2",
    "U",
    "U'",
    "U2",
    "D",
    "D'",
    "D2",
    "F",
    "F'",
    "F2",
    "B",
    "B'",
    "B2",
]

MOVE_TO_IDX = {move: idx for idx, move in enumerate(MOVES)}
NUM_MOVES = len(MOVES)

# Face indices
WHITE, YELLOW, RED, ORANGE, BLUE, GREEN = 0, 1, 2, 3, 4, 5
UP, DOWN, FRONT, BACK, RIGHT, LEFT = WHITE, YELLOW, RED, ORANGE, BLUE, GREEN

# Color names for display
COLOR_NAMES = ["W", "Y", "R", "O", "B", "G"]


class CubeState:
    """Represents the state of a Rubik's Cube."""

    def __init__(self, state: Optional[np.ndarray] = None):
        """
        Initialize cube state.

        Args:
            state: Optional 54-element array. If None, creates solved cube.
        """
        if state is not None:
            self.state = state.copy()
        else:
            # Solved cube: each face has 9 stickers of the same color
            self.state = np.array([face for face in range(6) for _ in range(9)], dtype=np.int8)

    def clone(self) -> "CubeState":
        """Create a copy of this cube state."""
        return CubeState(self.state)

    def is_solved(self) -> bool:
        """Check if the cube is in solved state."""
        for face in range(6):
            start = face * 9
            if not np.all(self.state[start : start + 9] == face):
                return False
        return True

    def get_face(self, face_idx: int) -> np.ndarray:
        """Get the 9 stickers of a face as a 3x3 array."""
        start = face_idx * 9
        return self.state[start : start + 9].reshape(3, 3)

    def set_face(self, face_idx: int, values: np.ndarray):
        """Set the 9 stickers of a face from a 3x3 or flat array."""
        start = face_idx * 9
        self.state[start : start + 9] = values.flatten()

    def _rotate_face_cw(self, face_idx: int):
        """Rotate a face 90 degrees clockwise."""
        face = self.get_face(face_idx)
        rotated = np.rot90(face, k=-1)  # k=-1 is clockwise
        self.set_face(face_idx, rotated)

    def _rotate_face_ccw(self, face_idx: int):
        """Rotate a face 90 degrees counter-clockwise."""
        face = self.get_face(face_idx)
        rotated = np.rot90(face, k=1)  # k=1 is counter-clockwise
        self.set_face(face_idx, rotated)

    def _rotate_face_180(self, face_idx: int):
        """Rotate a face 180 degrees."""
        face = self.get_face(face_idx)
        rotated = np.rot90(face, k=2)
        self.set_face(face_idx, rotated)

    def apply_move(self, move: str) -> "CubeState":
        """
        Apply a move to the cube and return self for chaining.

        Args:
            move: Move notation (R, R', R2, L, L', L2, etc.)

        Returns:
            self for method chaining
        """
        if move == "R":
            self._move_R()
        elif move == "R'":
            self._move_R_prime()
        elif move == "R2":
            self._move_R()
            self._move_R()
        elif move == "L":
            self._move_L()
        elif move == "L'":
            self._move_L_prime()
        elif move == "L2":
            self._move_L()
            self._move_L()
        elif move == "U":
            self._move_U()
        elif move == "U'":
            self._move_U_prime()
        elif move == "U2":
            self._move_U()
            self._move_U()
        elif move == "D":
            self._move_D()
        elif move == "D'":
            self._move_D_prime()
        elif move == "D2":
            self._move_D()
            self._move_D()
        elif move == "F":
            self._move_F()
        elif move == "F'":
            self._move_F_prime()
        elif move == "F2":
            self._move_F()
            self._move_F()
        elif move == "B":
            self._move_B()
        elif move == "B'":
            self._move_B_prime()
        elif move == "B2":
            self._move_B()
            self._move_B()
        else:
            raise ValueError(f"Unknown move: {move}")

        return self

    def apply_moves(self, moves: List[str]) -> "CubeState":
        """Apply a sequence of moves."""
        for move in moves:
            self.apply_move(move)
        return self

    def _move_R(self):
        """Right face clockwise."""
        self._rotate_face_cw(RIGHT)

        # Save the column that will be overwritten
        temp = self.state[UP * 9 + 2], self.state[UP * 9 + 5], self.state[UP * 9 + 8]

        # Up <- Front
        self.state[UP * 9 + 2] = self.state[FRONT * 9 + 2]
        self.state[UP * 9 + 5] = self.state[FRONT * 9 + 5]
        self.state[UP * 9 + 8] = self.state[FRONT * 9 + 8]

        # Front <- Down
        self.state[FRONT * 9 + 2] = self.state[DOWN * 9 + 2]
        self.state[FRONT * 9 + 5] = self.state[DOWN * 9 + 5]
        self.state[FRONT * 9 + 8] = self.state[DOWN * 9 + 8]

        # Down <- Back (reversed)
        self.state[DOWN * 9 + 2] = self.state[BACK * 9 + 6]
        self.state[DOWN * 9 + 5] = self.state[BACK * 9 + 3]
        self.state[DOWN * 9 + 8] = self.state[BACK * 9 + 0]

        # Back <- Up (reversed)
        self.state[BACK * 9 + 6] = temp[0]
        self.state[BACK * 9 + 3] = temp[1]
        self.state[BACK * 9 + 0] = temp[2]

    def _move_R_prime(self):
        """Right face counter-clockwise."""
        for _ in range(3):
            self._move_R()

    def _move_L(self):
        """Left face clockwise."""
        self._rotate_face_cw(LEFT)

        # Save the column
        temp = self.state[UP * 9 + 0], self.state[UP * 9 + 3], self.state[UP * 9 + 6]

        # Up <- Back (reversed)
        self.state[UP * 9 + 0] = self.state[BACK * 9 + 8]
        self.state[UP * 9 + 3] = self.state[BACK * 9 + 5]
        self.state[UP * 9 + 6] = self.state[BACK * 9 + 2]

        # Back <- Down (reversed)
        self.state[BACK * 9 + 8] = self.state[DOWN * 9 + 0]
        self.state[BACK * 9 + 5] = self.state[DOWN * 9 + 3]
        self.state[BACK * 9 + 2] = self.state[DOWN * 9 + 6]

        # Down <- Front
        self.state[DOWN * 9 + 0] = self.state[FRONT * 9 + 0]
        self.state[DOWN * 9 + 3] = self.state[FRONT * 9 + 3]
        self.state[DOWN * 9 + 6] = self.state[FRONT * 9 + 6]

        # Front <- Up
        self.state[FRONT * 9 + 0] = temp[0]
        self.state[FRONT * 9 + 3] = temp[1]
        self.state[FRONT * 9 + 6] = temp[2]

    def _move_L_prime(self):
        """Left face counter-clockwise."""
        for _ in range(3):
            self._move_L()

    def _move_U(self):
        """Up face clockwise."""
        self._rotate_face_cw(UP)

        # Save the row
        temp = self.state[FRONT * 9 + 0], self.state[FRONT * 9 + 1], self.state[FRONT * 9 + 2]

        # Front <- Right
        self.state[FRONT * 9 + 0] = self.state[RIGHT * 9 + 0]
        self.state[FRONT * 9 + 1] = self.state[RIGHT * 9 + 1]
        self.state[FRONT * 9 + 2] = self.state[RIGHT * 9 + 2]

        # Right <- Back
        self.state[RIGHT * 9 + 0] = self.state[BACK * 9 + 0]
        self.state[RIGHT * 9 + 1] = self.state[BACK * 9 + 1]
        self.state[RIGHT * 9 + 2] = self.state[BACK * 9 + 2]

        # Back <- Left
        self.state[BACK * 9 + 0] = self.state[LEFT * 9 + 0]
        self.state[BACK * 9 + 1] = self.state[LEFT * 9 + 1]
        self.state[BACK * 9 + 2] = self.state[LEFT * 9 + 2]

        # Left <- Front
        self.state[LEFT * 9 + 0] = temp[0]
        self.state[LEFT * 9 + 1] = temp[1]
        self.state[LEFT * 9 + 2] = temp[2]

    def _move_U_prime(self):
        """Up face counter-clockwise."""
        for _ in range(3):
            self._move_U()

    def _move_D(self):
        """Down face clockwise."""
        self._rotate_face_cw(DOWN)

        # Save the row
        temp = self.state[FRONT * 9 + 6], self.state[FRONT * 9 + 7], self.state[FRONT * 9 + 8]

        # Front <- Left
        self.state[FRONT * 9 + 6] = self.state[LEFT * 9 + 6]
        self.state[FRONT * 9 + 7] = self.state[LEFT * 9 + 7]
        self.state[FRONT * 9 + 8] = self.state[LEFT * 9 + 8]

        # Left <- Back
        self.state[LEFT * 9 + 6] = self.state[BACK * 9 + 6]
        self.state[LEFT * 9 + 7] = self.state[BACK * 9 + 7]
        self.state[LEFT * 9 + 8] = self.state[BACK * 9 + 8]

        # Back <- Right
        self.state[BACK * 9 + 6] = self.state[RIGHT * 9 + 6]
        self.state[BACK * 9 + 7] = self.state[RIGHT * 9 + 7]
        self.state[BACK * 9 + 8] = self.state[RIGHT * 9 + 8]

        # Right <- Front
        self.state[RIGHT * 9 + 6] = temp[0]
        self.state[RIGHT * 9 + 7] = temp[1]
        self.state[RIGHT * 9 + 8] = temp[2]

    def _move_D_prime(self):
        """Down face counter-clockwise."""
        for _ in range(3):
            self._move_D()

    def _move_F(self):
        """Front face clockwise."""
        self._rotate_face_cw(FRONT)

        # Save the row
        temp = self.state[UP * 9 + 6], self.state[UP * 9 + 7], self.state[UP * 9 + 8]

        # Up <- Left (rotated)
        self.state[UP * 9 + 6] = self.state[LEFT * 9 + 8]
        self.state[UP * 9 + 7] = self.state[LEFT * 9 + 5]
        self.state[UP * 9 + 8] = self.state[LEFT * 9 + 2]

        # Left <- Down
        self.state[LEFT * 9 + 2] = self.state[DOWN * 9 + 0]
        self.state[LEFT * 9 + 5] = self.state[DOWN * 9 + 1]
        self.state[LEFT * 9 + 8] = self.state[DOWN * 9 + 2]

        # Down <- Right (rotated)
        self.state[DOWN * 9 + 0] = self.state[RIGHT * 9 + 6]
        self.state[DOWN * 9 + 1] = self.state[RIGHT * 9 + 3]
        self.state[DOWN * 9 + 2] = self.state[RIGHT * 9 + 0]

        # Right <- Up
        self.state[RIGHT * 9 + 0] = temp[0]
        self.state[RIGHT * 9 + 3] = temp[1]
        self.state[RIGHT * 9 + 6] = temp[2]

    def _move_F_prime(self):
        """Front face counter-clockwise."""
        for _ in range(3):
            self._move_F()

    def _move_B(self):
        """Back face clockwise."""
        self._rotate_face_cw(BACK)

        # Save the row
        temp = self.state[UP * 9 + 0], self.state[UP * 9 + 1], self.state[UP * 9 + 2]

        # Up <- Right (rotated)
        self.state[UP * 9 + 0] = self.state[RIGHT * 9 + 2]
        self.state[UP * 9 + 1] = self.state[RIGHT * 9 + 5]
        self.state[UP * 9 + 2] = self.state[RIGHT * 9 + 8]

        # Right <- Down (rotated)
        self.state[RIGHT * 9 + 2] = self.state[DOWN * 9 + 8]
        self.state[RIGHT * 9 + 5] = self.state[DOWN * 9 + 7]
        self.state[RIGHT * 9 + 8] = self.state[DOWN * 9 + 6]

        # Down <- Left
        self.state[DOWN * 9 + 6] = self.state[LEFT * 9 + 0]
        self.state[DOWN * 9 + 7] = self.state[LEFT * 9 + 3]
        self.state[DOWN * 9 + 8] = self.state[LEFT * 9 + 6]

        # Left <- Up (rotated)
        self.state[LEFT * 9 + 0] = temp[2]
        self.state[LEFT * 9 + 3] = temp[1]
        self.state[LEFT * 9 + 6] = temp[0]

    def _move_B_prime(self):
        """Back face counter-clockwise."""
        for _ in range(3):
            self._move_B()

    def scramble(self, num_moves: int = 20) -> List[str]:
        """
        Scramble the cube with random moves.

        Args:
            num_moves: Number of random moves to apply

        Returns:
            List of moves applied
        """
        moves_applied = []
        last_move_base = None

        for _ in range(num_moves):
            # Avoid redundant moves (e.g., R followed by R')
            available_moves = MOVES.copy()
            if last_move_base:
                available_moves = [m for m in available_moves if not m.startswith(last_move_base)]

            move = random.choice(available_moves)
            self.apply_move(move)
            moves_applied.append(move)
            last_move_base = move[0]  # Get base move (R, L, U, etc.)

        return moves_applied

    def to_one_hot(self) -> np.ndarray:
        """
        Convert state to one-hot encoding for neural network input.

        Returns:
            Array of shape (324,) - 54 positions × 6 colors
        """
        one_hot = np.zeros((54, 6), dtype=np.float32)
        for i, color in enumerate(self.state):
            one_hot[i, color] = 1.0
        return one_hot.flatten()

    def count_correct_stickers(self) -> int:
        """Count number of stickers in their correct position (0-54)."""
        correct = 0
        for face in range(6):
            start = face * 9
            correct += np.sum(self.state[start : start + 9] == face)
        return int(correct)

    def count_correct_faces(self) -> int:
        """Count number of completely solved faces (0-6)."""
        correct = 0
        for face in range(6):
            start = face * 9
            if np.all(self.state[start : start + 9] == face):
                correct += 1
        return correct

    def __str__(self) -> str:
        """String representation of the cube (unfolded view)."""
        lines = []

        # Up face
        up = self.get_face(UP)
        for row in up:
            lines.append("      " + " ".join(COLOR_NAMES[c] for c in row))

        lines.append("")

        # Middle row: Left, Front, Right, Back
        for row_idx in range(3):
            row_str = ""
            for face_idx in [LEFT, FRONT, RIGHT, BACK]:
                face = self.get_face(face_idx)
                row_str += " ".join(COLOR_NAMES[c] for c in face[row_idx]) + "  "
            lines.append(row_str)

        lines.append("")

        # Down face
        down = self.get_face(DOWN)
        for row in down:
            lines.append("      " + " ".join(COLOR_NAMES[c] for c in row))

        return "\n".join(lines)

    def __eq__(self, other: "CubeState") -> bool:
        """Check equality with another cube state."""
        return np.array_equal(self.state, other.state)

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash(self.state.tobytes())


def get_inverse_move(move: str) -> str:
    """Get the inverse of a move."""
    if move.endswith("'"):
        return move[:-1]
    elif move.endswith("2"):
        return move  # 180° moves are self-inverse
    else:
        return move + "'"


def simplify_moves(moves: List[str]) -> List[str]:
    """Simplify a sequence of moves by canceling redundant ones."""
    if not moves:
        return []

    result = []
    for move in moves:
        if not result:
            result.append(move)
            continue

        last = result[-1]
        base = move[0]
        last_base = last[0]

        if base != last_base:
            result.append(move)
            continue

        # Same base move - combine or cancel
        inverse = get_inverse_move(last)
        if move == inverse:
            result.pop()  # Cancel out
        elif last.endswith("2") and move.endswith("2"):
            result.pop()  # Two 180° = nothing
        elif last.endswith("2") or move.endswith("2"):
            # One is 180°, combine
            result.pop()
            if move.endswith("2"):
                result.append(get_inverse_move(last))
            else:
                result.append(get_inverse_move(move))
        else:
            # Two single moves
            if move == last:
                result.pop()
                result.append(base + "2")
            else:
                result.pop()  # They cancel

    return result
