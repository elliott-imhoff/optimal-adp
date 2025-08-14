"""Command-line interface for ADP optimization."""

import argparse
import logging
import sys
from pathlib import Path

from optimal_adp.config import get_total_rounds, get_total_picks
from optimal_adp.optimizer import optimize_adp


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.

    Args:
        verbose: If True, enable DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Run the CLI for ADP optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize fantasy football Average Draft Position (ADP) using regret minimization"
    )

    # Required arguments
    parser.add_argument(
        "data_file", help="Path to CSV file containing player statistics"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--output",
        default="artifacts/final_adp.csv",
        help="Output path for final ADP results (default: artifacts/final_adp.csv)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of optimization iterations (default: 50)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for ADP updates (default: 0.1)",
    )

    # Draft configuration
    parser.add_argument(
        "--num-teams",
        type=int,
        default=10,
        help="Number of teams in the draft (default: 10)",
    )

    # Logging options
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose (DEBUG level) logging"
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
