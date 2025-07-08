"""Setup for model_langgraph module."""
from setuptools import setup, find_packages

setup(
    name="model_langgraph",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.5.0",
        "pydantic-settings>=2.0.0",
        "anthropic>=0.47.1",
        "openai>=1.70.2",
    ],
)