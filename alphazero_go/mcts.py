import math
import numpy as np
from typing import Dict, Optional, Tuple

try:
    from .board import GoBoard
    from .network import PolicyValueNet
except ImportError:
    from board import GoBoard
    from network import PolicyValueNet

class Node:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Tuple[int,int], Node] = {}
        self.is_expanded = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, board: GoBoard, priors: np.ndarray):
        self.is_expanded = True
        for move, p in zip(board.get_legal_moves(), priors):
            self.children[move] = Node(p)

    def select_child(self, c_puct: float = 1.0) -> Tuple[Tuple[int,int], 'Node']:
        total_visits = sum(child.visit_count for child in self.children.values())
        best_score = -float('inf')
        best_move = None
        best_child = None
        for move, child in self.children.items():
            prior_score = c_puct * child.prior * math.sqrt(total_visits + 1) / (1 + child.visit_count)
            value_score = child.value()
            score = value_score + prior_score
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

class MCTS:
    def __init__(self, model: PolicyValueNet, num_simulations: int = 50, c_puct: float = 1.0, verbose: bool = False):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.verbose = verbose

    def run(self, board: GoBoard) -> np.ndarray:
        root = Node(0)
        board_tensor = board.to_tensor()[None, :]
        policy, value = self.model.predict(board_tensor)
        legal_moves = board.get_legal_moves()
        priors = np.array([policy[0, m[0] * board.size + m[1]] for m in legal_moves])
        priors = priors / np.sum(priors)
        root.expand(board, priors)
        root.value_sum = value.item()
        root.visit_count = 1

        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.fast_copy()
            search_path = [node]
            # select
            while node.is_expanded and node.children:
                move, node = node.select_child(self.c_puct)
                scratch_board.make_move(move)
                search_path.append(node)
            # expand
            if not node.is_expanded:
                board_tensor = scratch_board.to_tensor()[None, :]
                policy, value = self.model.predict(board_tensor)
                legal_moves = scratch_board.get_legal_moves()
                if legal_moves:
                    priors = np.array([policy[0, m[0] * board.size + m[1]] for m in legal_moves])
                    priors = priors / np.sum(priors)
                    node.expand(scratch_board, priors)
                else:
                    value = float(scratch_board.result())
                node.value_sum += float(value if isinstance(value, float) else value.item())
                node.visit_count += 1
            # backprop
            value = node.value()
            for n in reversed(search_path[:-1]):
                n.value_sum += value
                n.visit_count += 1
                value = -value

        visits = np.zeros(board.size * board.size, dtype=np.float32)
        for move, child in root.children.items():
            idx = move[0] * board.size + move[1]
            visits[idx] = child.visit_count
        if np.sum(visits) > 0:
            visits /= np.sum(visits)
        
        if self.verbose:
            self._print_search_stats(root, board)
        
        return visits.reshape(board.size, board.size)
    
    def _print_search_stats(self, root: Node, board: GoBoard):
        print(f"MCTS Search Statistics (after {self.num_simulations} simulations):")
        print(f"Root visits: {root.visit_count}, Root value: {root.value():.3f}")
        
        # Sort moves by visit count
        sorted_moves = sorted(root.children.items(), key=lambda x: x[1].visit_count, reverse=True)
        
        print("Top moves:")
        for i, (move, child) in enumerate(sorted_moves[:5]):
            x, y = move
            print(f"  {i+1}. ({x},{y}): visits={child.visit_count:3d}, value={child.value():+.3f}, prior={child.prior:.3f}")
        
        # Show visit count heatmap
        print("\nVisit count heatmap:")
        visit_grid = np.zeros((board.size, board.size), dtype=int)
        for move, child in root.children.items():
            visit_grid[move[0], move[1]] = child.visit_count
        
        for y in range(board.size):
            row = []
            for x in range(board.size):
                if visit_grid[x, y] > 0:
                    row.append(f"{visit_grid[x, y]:3d}")
                else:
                    row.append("  .")
            print("  " + " ".join(row))
        print()
