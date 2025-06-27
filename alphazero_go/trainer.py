import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

try:
    from .board import GoBoard
    from .mcts import MCTS
    from .network import PolicyValueNet
    from .replay_buffer import ReplayBuffer
except ImportError:
    from board import GoBoard
    from mcts import MCTS
    from network import PolicyValueNet
    from replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self, board_size: int = 5, buffer_size: int = 10000, verbose: bool = False, show_heatmaps: bool = False):
        self.board_size = board_size
        self.model = PolicyValueNet(board_size)
        self.buffer = ReplayBuffer(buffer_size)
        # Aggressive MCTS reduction for tiny boards
        if board_size <= 3:
            num_sims = 3  # Minimal simulations for 3x3
        elif board_size <= 4:
            num_sims = 5  # Very few for 4x4
        else:
            num_sims = max(5, min(15, board_size * board_size // 3))
        
        if verbose:
            print(f"Using {num_sims} MCTS simulations for {board_size}x{board_size} board")
        
        self.mcts = MCTS(self.model, num_simulations=num_sims, verbose=verbose)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.verbose = verbose
        self.show_heatmaps = show_heatmaps

    def self_play(self, num_games: int = 1):
        for game_num in range(num_games):
            if self.verbose:
                print(f"\n=== Self-Play Game {game_num + 1}/{num_games} ===")
            
            board = GoBoard(self.board_size)
            states = []
            mcts_probs = []
            players = []
            move_count = 0
            
            while not board.is_game_over():
                move_count += 1
                probs = self.mcts.run(board)
                
                if self.verbose:
                    print(f"\nMove {move_count} (Player {'Black' if board.current_player == 1 else 'White'}):")
                    if self.show_heatmaps:
                        board.display_probability_heatmap(probs)
                    else:
                        board.display_with_probabilities(probs)
                move_probs = probs.flatten()
                legal_moves = board.get_legal_moves()
                if not legal_moves:
                    board.make_move(None)
                    if self.verbose:
                        print("Pass move (no legal moves)")
                    continue
                probs_list = [move_probs[m[0] * board.size + m[1]] for m in legal_moves]
                probs_list = np.array(probs_list)
                probs_list = probs_list / np.sum(probs_list)
                move = legal_moves[np.random.choice(len(legal_moves), p=probs_list)]
                
                if self.verbose:
                    print(f"Selected move: ({move[0]}, {move[1]})")
                
                states.append(board.to_tensor())
                mcts_probs.append(move_probs)
                players.append(board.current_player)
                board.make_move(move)
            
            winner = board.result()
            if self.verbose:
                print(f"\nFinal board:")
                print(board)
                winner_name = "Black" if winner == 1 else "White" if winner == -1 else "Draw"
                print(f"Game result: {winner_name} wins" if winner != 0 else "Game result: Draw")
            
            for s, p, player in zip(states, mcts_probs, players):
                value = winner if player == winner else -winner
                self.buffer.add(s, p, value)

    def train_step(self, batch_size: int = 32):
        if len(self.buffer) < batch_size:
            return None
        states, policies, values = self.buffer.sample(batch_size)
        self.model.train()
        self.optimizer.zero_grad()
        policy_pred, value_pred = self.model(torch.from_numpy(states))
        policy_loss = -torch.mean(torch.sum(torch.from_numpy(policies) * F.log_softmax(policy_pred, dim=1), dim=1))
        value_loss = F.mse_loss(value_pred.squeeze(), torch.from_numpy(values))
        loss = policy_loss + value_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, iterations: int = 10, games_per_iter: int = 2, batch_size: int = 32):
        for i in range(iterations):
            print(f"\n{'='*50}")
            print(f"Training Iteration {i+1}/{iterations}")
            print(f"{'='*50}")
            
            self.self_play(games_per_iter)
            loss = self.train_step(batch_size)
            print(f"Iteration {i+1} complete - Loss: {loss}")
            
            # Show periodic heatmap demonstration
            if self.show_heatmaps and (i + 1) % max(1, iterations // 3) == 0:
                print(f"\n--- Heatmap Demo After Iteration {i+1} ---")
                self._demo_heatmap()
    
    def _demo_heatmap(self):
        """Show a demonstration heatmap on a fresh board."""
        demo_board = GoBoard(self.board_size)
        
        # Add a few random moves to make it interesting
        import random
        moves_to_make = min(3, self.board_size * self.board_size // 3)
        for _ in range(moves_to_make):
            legal_moves = demo_board.get_legal_moves()
            if legal_moves:
                move = random.choice(legal_moves)
                demo_board.make_move(move)
        
        print("Demonstrating MCTS analysis on current training position:")
        # Temporarily disable verbose for clean demo
        old_verbose = self.mcts.verbose
        self.mcts.verbose = False
        probs = self.mcts.run(demo_board)
        self.mcts.verbose = old_verbose
        
        demo_board.display_probability_heatmap(probs)
        print("--- End Heatmap Demo ---\n")

