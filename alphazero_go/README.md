# Alphazero Go (Simplified)

This folder contains a minimal implementation of a Go environment and a training
pipeline inspired by AlphaZero. It is intended for small board sizes (default 5x5)
and can be trained on a laptop.

## Components
- `board.py` – a minimal Go board with capture rules.
- `mcts.py` – Monte Carlo Tree Search using a neural network for policy and value
  predictions.
- `network.py` – PyTorch model producing move probabilities and a state value.
- `replay_buffer.py` – a simple replay buffer for storing self-play data.
- `trainer.py` – combines MCTS and the network to generate self-play games and
  update the model.
- `play.py` – command line interface to train or play using a saved model.

## Usage

Install required packages (PyTorch and NumPy) and run training:
```bash
pip install torch numpy
python -m alphazero_go.play train --iters 5 --games 2 --size 5
```
After training, play a game using the trained model:
```bash
python -m alphazero_go.play play --model model.pt --size 5
```
