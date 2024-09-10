import argparse
from tac import run_tac
from handwriting import run_handwriting
from mindlamp import run_regression, run_classification

def main():
    parser = argparse.ArgumentParser(description="Run different models with arguments")

    subparsers = parser.add_subparsers(dest='command', help='Sub-command to run specific tasks')

    # Handwriting arguments
    handwriting_parser = subparsers.add_parser('handwriting', help='Run handwriting model')
    handwriting_parser.add_argument('--batch', type=int, default=32, help='Batch size')
    handwriting_parser.add_argument('--epochs', type=int, default=12, help='Number of epochs')
    handwriting_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    handwriting_parser.add_argument('--simple', type=bool, default=False, help='Use simple model')

    # TAC arguments
    tac_parser = subparsers.add_parser('tac', help='Run TAC model')
    tac_parser.add_argument('--seq_len', type=int, default=1000, help='Sequence length')
    tac_parser.add_argument('--batch', type=int, default=32, help='Batch size')
    tac_parser.add_argument('--epochs', type=int, default=12, help='Number of epochs')
    tac_parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    tac_parser.add_argument('--simple', type=bool, default=False, help='Use simple model')

    # Regression arguments
    regression_parser = subparsers.add_parser('mindlamp_r', help='Run mindlamp regression model')
    regression_parser.add_argument('--target', type=str, default='PHQ-9', help='Target variable')
    regression_parser.add_argument('--batch', type=int, default=8, help='Batch size')
    regression_parser.add_argument('--seq_len', type=int, default=6000, help='Sequence length')
    regression_parser.add_argument('--epochs', type=int, default=12, help='Number of epochs')
    regression_parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    regression_parser.add_argument('--simple', type=bool, default=False, help='Use simple model')

    # Classification arguments
    classification_parser = subparsers.add_parser('mindlamp_c', help='Run mindlamp classification model')
    classification_parser.add_argument('--target', type=str, default='is_PHQ-9', help='Target variable')
    classification_parser.add_argument('--batch', type=int, default=8, help='Batch size')
    classification_parser.add_argument('--seq_len', type=int, default=6000, help='Sequence length')
    classification_parser.add_argument('--epochs', type=int, default=12, help='Number of epochs')
    classification_parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    classification_parser.add_argument('--simple', type=bool, default=False, help='Use simple model')

    args = parser.parse_args()

    if args.command == 'handwriting':
        run_handwriting(batch=args.batch, epochs=args.epochs, lr=args.lr, simple=args.simple)
    elif args.command == 'tac':
        run_tac(seq_len=args.seq_len, batch=args.batch, epochs=args.epochs, lr=args.lr, simple=args.simple)
    elif args.command == 'mindlamp_r':
        run_regression(target=args.target, batch=args.batch, seq_len=args.seq_len, epochs=args.epochs, lr=args.lr, simple=args.simple)
    elif args.command == 'mindlamp_c':
        run_classification(target=args.target, batch=args.batch, seq_len=args.seq_len, epochs=args.epochs, lr=args.lr, simple=args.simple)
    else:
        print("Please provide a valid command. Use --help for more details.")

if __name__ == "__main__":
    main()