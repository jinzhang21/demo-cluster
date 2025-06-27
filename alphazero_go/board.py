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
    
    def display_with_probabilities(self, move_probs: np.ndarray):
        """Display board with move probabilities as percentages."""
        chars = {1: 'X', -1: 'O', 0: '.'}
        
        # Board display
        print("Current board:")
        for y in range(self.size):
            row = ''.join(chars[self.board[x, y]] for x in range(self.size))
            print(f"  {row}")
        
        # Probability display
        print("\nMove probabilities (%):")
        prob_grid = move_probs.reshape(self.size, self.size) if len(move_probs.shape) == 1 else move_probs
        
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if self.board[x, y] == 0:  # Empty position
                    prob_pct = prob_grid[x, y] * 100
                    if prob_pct >= 10:
                        row.append(f"{prob_pct:2.0f}")
                    elif prob_pct >= 1:
                        row.append(f"{prob_pct:2.1f}")
                    else:
                        row.append(f"{prob_pct:2.2f}"[:2])
                else:  # Occupied position
                    row.append(" -")
            print("  " + " ".join(row))
        print()
    
    def display_probability_heatmap(self, move_probs: np.ndarray):
        """Display MCTS probability heatmap with visual intensity."""
        chars = {1: 'X', -1: 'O', 0: '.'}
        prob_grid = move_probs.reshape(self.size, self.size) if len(move_probs.shape) == 1 else move_probs
        
        # Board with pieces
        print("Board state:")
        for y in range(self.size):
            row = ''.join(chars[self.board[x, y]] for x in range(self.size))
            print(f"  {row}")
        
        # Probability heatmap with visual intensity
        print("\nMCTS Probability Heatmap:")
        max_prob = np.max(prob_grid) if np.max(prob_grid) > 0 else 1
        
        # Header with column indices
        print("    " + " ".join(f"{i:2d}" for i in range(self.size)))
        print("  +" + "---" * self.size + "+")
        
        for y in range(self.size):
            row_str = f"{y:2d}|"
            for x in range(self.size):
                if self.board[x, y] != 0:  # Occupied position
                    piece = chars[self.board[x, y]]
                    row_str += f" {piece} "
                else:  # Empty position - show probability intensity
                    prob = prob_grid[x, y]
                    intensity = prob / max_prob if max_prob > 0 else 0
                    prob_pct = prob * 100
                    
                    # Visual intensity using different characters
                    if intensity >= 0.8:
                        symbol = "██"  # Very high
                    elif intensity >= 0.6:
                        symbol = "▓▓"  # High
                    elif intensity >= 0.4:
                        symbol = "▒▒"  # Medium
                    elif intensity >= 0.2:
                        symbol = "░░"  # Low
                    elif intensity >= 0.05:
                        symbol = "··"  # Very low
                    else:
                        symbol = "  "  # Negligible
                    
                    row_str += f"{symbol}"
            row_str += "|"
            print(row_str)
        
        print("  +" + "---" * self.size + "+")
        
        # Legend
        print("\nLegend: ██ Very High (80%+)  ▓▓ High (60%+)  ▒▒ Med (40%+)  ░░ Low (20%+)  ·· Very Low (5%+)")
        
        # Show actual percentages for positions with significant probability
        print("\nSignificant move probabilities:")
        significant_moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == 0 and prob_grid[x, y] >= 0.05:  # 5% threshold
                    significant_moves.append((prob_grid[x, y], x, y))
        
        significant_moves.sort(reverse=True)
        for prob, x, y in significant_moves[:8]:  # Show top 8
            print(f"  ({x},{y}): {prob*100:5.1f}%")
        
        print()
