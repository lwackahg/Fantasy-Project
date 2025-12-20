# 2. Core Principles & Style Guide

This document outlines the development philosophy and coding standards for the Fantasy Trade Analyzer project. Adhering to these guidelines ensures the codebase remains clean, consistent, and maintainable.

## 2.1. Core Programming Principles

-   **Clean and Readable Code**: Write code that is easy to understand. Use clear variable names, logical structure, and keep functions focused on a single task.
-   **Modularity and Reusability**: Design components and functions to be modular and reusable across different parts of the application. Avoid code duplication.
-   **Simplicity (KISS)**: Keep It Simple, Stupid. Favor simple, straightforward solutions over complex ones whenever possible.
-   **Robust Error Handling**: Implement comprehensive error handling to manage unexpected states, invalid inputs, and external service failures gracefully.
-   **Efficiency**: Write efficient algorithms and queries to ensure the application is responsive, especially during data-intensive calculations.
-   **Testability**: Write code with testing in mind. Unit tests are crucial for verifying functionality and preventing regressions.
-   **Secure Coding**: Be mindful of security best practices, especially when handling user data or external integrations.

## 2.2. Code Style Guide

### 2.2.1. Python (Backend)

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code with a few specific conventions:

-   **Indentation**: Use tabs for indentation.
-   **Line Length**: Maximum line length is 100 characters.
-   **Naming Conventions**:
    -   `snake_case` for variables and functions.
    -   `PascalCase` for classes.
    -   `CAPITALIZED_SNAKE_CASE` for constants.
-   **Comments**: Use clear, concise comments to explain complex logic. Docstrings should follow the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
-   **Type Hinting**: Use type hints for all function signatures to improve code clarity and allow for static analysis.
-   **Formatting**: Use an automated code formatter like Black or Ruff to ensure consistent style.

### 2.2.2. General

-   **File Naming**: Use `snake_case` for all file names (e.g., `schedule_analysis.py`).
-   **Commit Messages**: Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps in creating a clear and automated changelog. Example: `feat: add smart auction bot for bidding advice`.
