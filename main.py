"""Main entry point for the Bygningsreglementet Chat Bot."""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.config import Config
from src.utils.logging import setup_logging
from src.ui import create_app


def main():
    """Main function to run the application."""
    # Setup logging
    setup_logging(
        level="INFO" if not Config.DEBUG else "DEBUG",
        log_file=str(Config.BASE_DIR / "logs" / "app.log")
    )
    
    # Create and run the app
    app = create_app()
    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT
    )


if __name__ == "__main__":
    main()
