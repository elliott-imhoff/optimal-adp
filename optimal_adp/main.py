"""Command-line interface for ADP optimization."""

import argparse
import logging
import sys
from pathlib import Path

from optimal_adp.config import get_total_rounds, get_total_picks
from optimal_adp.optimizer import optimize_adp
from optimal_adp.validation import validate_optimization


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and emojis for different log levels."""

    LEVEL_COLORS = {
        "DEBUG": Colors.BLUE,
        "INFO": "",  # No color - plain white/default
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "CRITICAL": Colors.RED,
    }

    LEVEL_EMOJIS = {
        "DEBUG": "🔍 ",
        "INFO": "",  # No emoji for info messages
        "WARNING": "⚠️  ",
        "ERROR": "❌ ",
        "CRITICAL": "💥 ",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and emojis."""
        # Get color and emoji for this level
        level_color = self.LEVEL_COLORS.get(record.levelname, "")
        level_emoji = self.LEVEL_EMOJIS.get(record.levelname, "")

        # Format the message with color and emoji (if any)
        message = record.getMessage()
        if level_color:
            formatted_message = f"{level_emoji}{level_color}{message}{Colors.RESET}"
        else:
            formatted_message = f"{level_emoji}{message}"

        return formatted_message


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application with colors and emojis.

    Args:
        verbose: If True, enable DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Create custom handler with colored formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(handler)


def cmd_optimize(args: argparse.Namespace) -> None:
    """Handle the optimize subcommand."""
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate input file exists
    data_file_path = Path(args.data_file)
    if not data_file_path.exists():
        logger.error(f"Input file does not exist: {data_file_path}")
        sys.exit(1)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get draft configuration values
    num_teams = args.num_teams

    logger.info("Starting ADP optimization with configuration:")
    logger.info(f"  Input file: {data_file_path}")
    logger.info(f"  Output file: {output_path}")
    logger.info(
        f"  Draft config: {num_teams} teams, "
        f"{get_total_rounds()} rounds, {get_total_picks(num_teams)} total picks"
    )

    try:
        # Run optimization
        final_adp, convergence_history, iterations = optimize_adp(
            data_file_path=str(data_file_path),
            num_teams=num_teams,
            max_iterations=args.max_iterations,
            learning_rate=args.learning_rate,
            output_file_path=str(output_path),
        )

        # Print summary
        logger.info("Optimization completed successfully!")
        logger.info(f"Final ADP computed for {len(final_adp)} players")
        logger.info(f"Iterations completed: {iterations}")
        logger.info(f"Convergence history: {convergence_history}")

        if convergence_history and convergence_history[-1] == 0:
            logger.info("✅ Convergence achieved!")
        else:
            logger.info("⚠️  Maximum iterations reached without full convergence")

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if args.verbose:
            logger.exception("Full error traceback:")
        sys.exit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    """Handle the validate subcommand."""
    # Setup logging
    setup_logging(args.verbose)

    # Validate input file exists
    data_file_path = Path(args.data_file)
    if not data_file_path.exists():
        print(f"❌ Input file does not exist: {data_file_path}")
        sys.exit(1)

    # Run validation
    success = validate_optimization(
        data_file_path=str(data_file_path),
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        num_teams=args.num_teams,
        enable_perturbation=args.perturb,
        perturbation_factor=args.perturbation_factor,
        artifacts_outputs=not args.no_artifacts,
    )

    sys.exit(0 if success else 1)


def main() -> None:
    """Run the CLI for ADP optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize fantasy football Average Draft Position (ADP) using regret minimization"
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Optimize subcommand
    optimize_parser = subparsers.add_parser("optimize", help="Run ADP optimization")

    # Required arguments for optimize
    optimize_parser.add_argument(
        "data_file", help="Path to CSV file containing player statistics"
    )

    # Optional arguments for optimize
    optimize_parser.add_argument(
        "--output",
        default="artifacts/final_adp.csv",
        help="Output path for final ADP results (default: artifacts/final_adp.csv)",
    )

    optimize_parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum number of optimization iterations (default: 1000)",
    )

    optimize_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for ADP updates (default: 0.1)",
    )

    optimize_parser.add_argument(
        "--num-teams",
        type=int,
        default=10,
        help="Number of teams in the draft (default: 10)",
    )

    optimize_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose (DEBUG level) logging"
    )

    optimize_parser.set_defaults(func=cmd_optimize)

    # Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate", help="Validate optimization quality"
    )

    # Required arguments for validate
    validate_parser.add_argument(
        "data_file", help="Path to CSV file containing player statistics"
    )

    # Optional arguments for validate
    validate_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for optimization (default: 0.1)",
    )

    validate_parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum number of optimization iterations (default: 1000)",
    )

    validate_parser.add_argument(
        "--num-teams",
        type=int,
        default=10,
        help="Number of teams in the draft (default: 10)",
    )

    validate_parser.add_argument(
        "--perturb",
        action="store_true",
        help="Apply random perturbation to initial ADP values",
    )

    validate_parser.add_argument(
        "--perturbation-factor",
        type=float,
        default=0.1,
        help="Perturbation factor for initial ADP (default: 0.1 = 10 percent)",
    )

    validate_parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Disable archiving of optimization outputs",
    )

    validate_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose (DEBUG level) logging"
    )

    validate_parser.set_defaults(func=cmd_validate)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
