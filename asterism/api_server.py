#!/usr/bin/env python3
"""Entry point for the Asterism API server."""

import logging

import uvicorn

from asterism.config import Config


def main():
    """Run the API server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config = Config()

    # Run server
    uvicorn.run(
        "asterism.api:create_api_app",
        host=config.data.api.host,
        port=config.data.api.port,
        reload=config.data.api.debug,
        factory=True,
    )


if __name__ == "__main__":
    main()
