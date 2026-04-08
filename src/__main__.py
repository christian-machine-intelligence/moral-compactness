"""CLI entry point: python -m src <command>"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src {generate|score|analyze}")
        print()
        print("Commands:")
        print("  generate  Run scheming trials")
        print("  score     Score trials with judge model")
        print("  analyze   Statistical analysis of scored trials")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # strip command from argv

    if command == "generate":
        from .generate_data import main as run
        run()
    elif command == "score":
        from .score import main as run
        run()
    elif command == "analyze":
        from .analysis import main as run
        run()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
