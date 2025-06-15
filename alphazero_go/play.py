import argparse
import torch

from .trainer import Trainer
from .board import GoBoard
from .network import PolicyValueNet
from .mcts import MCTS


def train(args):
    trainer = Trainer(board_size=args.size)
    trainer.train(iterations=args.iters, games_per_iter=args.games, batch_size=args.batch)
    torch.save(trainer.model.state_dict(), args.model)


def play(args):
    board = GoBoard(args.size)
    model = PolicyValueNet(args.size)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    mcts = MCTS(model, num_simulations=50)
    while not board.is_game_over():
        probs = mcts.run(board)
        move_probs = probs.flatten()
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            board.make_move(None)
            continue
        best_move = max(legal_moves, key=lambda m: move_probs[m[0]*board.size + m[1]])
        board.make_move(best_move)
        print(board)
        print()
    print('Winner:', board.result())


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    t = subparsers.add_parser('train')
    t.add_argument('--size', type=int, default=5)
    t.add_argument('--iters', type=int, default=10)
    t.add_argument('--games', type=int, default=2)
    t.add_argument('--batch', type=int, default=32)
    t.add_argument('--model', type=str, default='model.pt')

    p = subparsers.add_parser('play')
    p.add_argument('--size', type=int, default=5)
    p.add_argument('--model', type=str, default='model.pt')

    args = parser.parse_args()
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'play':
        play(args)

if __name__ == '__main__':
    main()
