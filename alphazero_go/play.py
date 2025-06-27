import argparse
import torch

try:
    from .trainer import Trainer
    from .board import GoBoard
    from .network import PolicyValueNet
    from .mcts import MCTS
except ImportError:
    from trainer import Trainer
    from board import GoBoard
    from network import PolicyValueNet
    from mcts import MCTS


def train(args):
    show_heatmaps = getattr(args, 'heatmaps', False) or getattr(args, 'verbose', False)
    trainer = Trainer(board_size=args.size, verbose=args.verbose, show_heatmaps=show_heatmaps)
    trainer.train(iterations=args.iters, games_per_iter=args.games, batch_size=args.batch)
    torch.save(trainer.model.state_dict(), args.model)


def play(args):
    board = GoBoard(args.size)
    model = PolicyValueNet(args.size)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    verbose = getattr(args, 'verbose', False)
    show_heatmaps = getattr(args, 'heatmaps', False)
    mcts = MCTS(model, num_simulations=50, verbose=verbose)
    
    move_count = 0
    while not board.is_game_over():
        move_count += 1
        probs = mcts.run(board)
        
        if verbose or show_heatmaps:
            print(f"\nMove {move_count} (Player {'Black' if board.current_player == 1 else 'White'}):")
            if show_heatmaps:
                board.display_probability_heatmap(probs)
            else:
                board.display_with_probabilities(probs)
        else:
            print(board)
            print()
        
        move_probs = probs.flatten()
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            board.make_move(None)
            continue
        best_move = max(legal_moves, key=lambda m: move_probs[m[0]*board.size + m[1]])
        
        if verbose or show_heatmaps:
            print(f"AI selected move: ({best_move[0]}, {best_move[1]})")
        
        board.make_move(best_move)
    
    print(f"\nFinal board:")
    print(board)
    winner_name = "Black" if board.result() == 1 else "White" if board.result() == -1 else "Draw"
    print(f'Winner: {winner_name}')


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    t = subparsers.add_parser('train')
    t.add_argument('--size', type=int, default=5)
    t.add_argument('--iters', type=int, default=10)
    t.add_argument('--games', type=int, default=2)
    t.add_argument('--batch', type=int, default=32)
    t.add_argument('--model', type=str, default='model.pt')
    t.add_argument('--verbose', action='store_true', help='Show detailed training progress')
    t.add_argument('--heatmaps', action='store_true', help='Show MCTS probability heatmaps during training')

    p = subparsers.add_parser('play')
    p.add_argument('--size', type=int, default=5)
    p.add_argument('--model', type=str, default='model.pt')
    p.add_argument('--verbose', action='store_true', help='Show search statistics and move probabilities')
    p.add_argument('--heatmaps', action='store_true', help='Show MCTS probability heatmaps during play')

    args = parser.parse_args()
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'play':
        play(args)

if __name__ == '__main__':
    main()
