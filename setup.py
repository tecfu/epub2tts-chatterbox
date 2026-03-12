# setup.py is now handled by pyproject.toml
# This file is kept for backwards compatibility
from setuptools import setup, find_packages

setup(
    name="epub2tts-chatterbox",
    packages=find_packages(),
)
