# 5. Development Workflow

This document outlines the standard workflow for contributing to the Fantasy Trade Analyzer project. Following these steps ensures a smooth and consistent development process.

## 5.1. Version Control with Git

-   **Branching Strategy**: We use a simplified GitFlow model.
    -   `main`: This branch always contains the latest stable, production-ready code.
    -   `develop`: This is the primary development branch. All feature branches are merged into `develop`.
    -   `feat/<feature-name>`: All new features are developed in their own feature branch, created from `develop`.
    -   `fix/<issue-name>`: Bug fixes are handled in their own fix branch, created from `develop`.

-   **Pull Requests (PRs)**: All changes must be submitted via a pull request to the `develop` branch. PRs should be small, focused, and include a clear description of the changes.

-   **Code Reviews**: Every PR must be reviewed and approved by at least one other team member before it can be merged. Reviews should focus on code quality, correctness, and adherence to project standards.

## 5.2. Testing

-   **Unit Tests**: All new business logic in the `logic/` directory must be accompanied by unit tests. We use `pytest` as our testing framework.
-   **Running Tests**: Tests can be run from the root directory using the `pytest` command.
-   **Continuous Integration (CI)**: A CI pipeline (e.g., using GitHub Actions) should be set up to automatically run tests on every pull request.

## 5.3. Dependency Management

-   Project dependencies are managed in a `requirements.txt` file. 
-   When adding or updating a dependency, always update the `requirements.txt` file and include it in your commit.

## 5.4. Deployment

-   The application is deployed as a Streamlit app.
-   Deployment is done by merging the `develop` branch into the `main` branch after thorough testing.
-   The production environment should be configured to automatically deploy the latest version from the `main` branch.
