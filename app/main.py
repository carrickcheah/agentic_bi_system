"""
Minimal FastAPI Application Launcher

This module provides a clean entry point for the Agentic SQL Backend.
All application configuration is handled in api/app_factory.py
"""

from .api.app_factory import create_app
from .config import settings

# Create the app instance
app = create_app()


def main():
    """Run the application."""
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()