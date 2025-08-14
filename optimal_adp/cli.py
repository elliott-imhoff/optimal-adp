"""Command-line interface for ADP optimization."""

import logging
import sys
from pathlib import Path

import click

from optimal_adp.optimizer import run_optimization_loop


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


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option(
    "--learning-rate",
    type=float,
    default=0.1,
    help="Learning rate for optimization (default: 0.1)",
)
@click.option(
    "--max-iterations",
    type=int,
    default=1000,
    help="Maximum number of optimization iterations (default: 1000)",
)
@click.option(
    "--num-teams",
    type=int,
    default=10,
    help="Number of teams in the draft (default: 10)",
)
@click.option(
    "--perturb",
    type=float,
    default=0.0,
    help="Perturbation factor for initial ADP (default: 0.0 = no perturbation)",
)
@click.option(
    "--no-artifacts",
    is_flag=True,
    help="Disable archiving of optimization outputs",
)
@click.option("--verbose", is_flag=True, help="Enable verbose (DEBUG level) logging")
def run(
    data_file: str,
    learning_rate: float,
    max_iterations: int,
    num_teams: int,
    perturb: float,
    no_artifacts: bool,
    verbose: bool,
) -> None:
    """Optimize fantasy football Average Draft Position (ADP) using regret minimization."""
    # Setup logging
    setup_logging(verbose)

    # Validate input file exists
    data_file_path = Path(data_file)
    if not data_file_path.exists():
        print(f"❌ Input file does not exist: {data_file_path}")
        sys.exit(1)

    # Run optimization
    success = run_optimization_loop(
        data_file_path=str(data_file_path),
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        num_teams=num_teams,
        perturbation_factor=perturb,
        artifacts_outputs=not no_artifacts,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    run()
