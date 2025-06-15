import numpy as np
from typing import List, Tuple

class GoBoard:
    """Simple Go board for small-size boards. Uses 1 for black, -1 for white."""

    def __init__(self, size: int = 5):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1
        self.passes = 0

    def copy(self) -> 'GoBoard':
        new = GoBoard(self.size)
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.passes = self.passes
        return new

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        return [(x+dx, y+dy) for dx, dy in dirs if self.in_bounds(x+dx, y+dy)]

    def _get_group(self, x: int, y: int) -> Tuple[List[Tuple[int,int]], int]:
        color = self.board[x, y]
        stack = [(x, y)]
        visited = set(stack)
        group = []
        liberties = 0
        while stack:
            cx, cy = stack.pop()
            group.append((cx, cy))
            for nx, ny in self.neighbors(cx, cy):
                if self.board[nx, ny] == 0:
                    liberties += 1
                elif self.board[nx, ny] == color and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
        return group, liberties

    def _remove_dead_stones(self, color: int):
        removed = False
        visited = set()
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == color and (x, y) not in visited:
                    group, liberties = self._get_group(x, y)
                    visited.update(group)
                    if liberties == 0:
                        removed = True
                        for gx, gy in group:
                            self.board[gx, gy] = 0
        return removed

    def get_legal_moves(self) -> List[Tuple[int,int]]:
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == 0:
                    test_board = self.copy()
                    if test_board.make_move((x, y), check_only=True):
                        moves.append((x, y))
        return moves

    def make_move(self, move: Tuple[int,int], check_only: bool=False) -> bool:
        if move is None:
            self.passes += 1
            self.current_player *= -1
            return True
        x, y = move
        if not self.in_bounds(x, y) or self.board[x, y] != 0:
            return False
        self.board[x, y] = self.current_player
        # capture opponent stones
        self._remove_dead_stones(-self.current_player)
        # suicide check
        group, liberties = self._get_group(x, y)
        if liberties == 0:
            # illegal move
            self.board[x, y] = 0
            return False
        if not check_only:
            self.current_player *= -1
            self.passes = 0
        return True

    def is_game_over(self) -> bool:
        return self.passes >= 2

    def result(self) -> int:
        """Return the winner: 1 if black wins, -1 if white wins, 0 for tie."""
        black = np.sum(self.board == 1)
        white = np.sum(self.board == -1)
        if black > white:
            return 1
        elif white > black:
            return -1
        return 0

    def to_tensor(self) -> np.ndarray:
        """Return board tensor with shape (2, size, size)."""
        black = (self.board == 1).astype(np.float32)
        white = (self.board == -1).astype(np.float32)
        return np.stack([black, white], axis=0)

    def __str__(self):
        chars = {1: 'X', -1: 'O', 0: '.'}
        rows = []
        for y in range(self.size):
            row = ''.join(chars[self.board[x, y]] for x in range(self.size))
            rows.append(row)
        return '\n'.join(rows)
