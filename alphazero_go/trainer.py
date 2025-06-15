import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .board import GoBoard
from .mcts import MCTS
from .network import PolicyValueNet
from .replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self, board_size: int = 5, buffer_size: int = 10000):
        self.board_size = board_size
        self.model = PolicyValueNet(board_size)
        self.buffer = ReplayBuffer(buffer_size)
        self.mcts = MCTS(self.model, num_simulations=30)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def self_play(self, num_games: int = 1):
        for _ in range(num_games):
            board = GoBoard(self.board_size)
            states = []
            mcts_probs = []
            players = []
            while not board.is_game_over():
                probs = self.mcts.run(board)
                move_probs = probs.flatten()
                legal_moves = board.get_legal_moves()
                if not legal_moves:
                    board.make_move(None)
                    continue
                probs_list = [move_probs[m[0] * board.size + m[1]] for m in legal_moves]
                probs_list = np.array(probs_list)
                probs_list = probs_list / np.sum(probs_list)
                move = legal_moves[np.random.choice(len(legal_moves), p=probs_list)]
                states.append(board.to_tensor())
                mcts_probs.append(move_probs)
                players.append(board.current_player)
                board.make_move(move)
            winner = board.result()
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
            self.self_play(games_per_iter)
            loss = self.train_step(batch_size)
            print(f"Iter {i}, loss {loss}")

