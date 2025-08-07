# 1. Project Overview: Fantasy Trade Analyzer

## 1.1. Introduction

The Fantasy Trade Analyzer is a powerful Streamlit-based web application designed to give fantasy basketball managers a competitive edge. It provides a suite of advanced tools for in-depth analysis of trades, schedules, and auction drafts, moving beyond simple player comparisons to offer data-driven, strategic insights.

## 1.2. Purpose and Vision

The primary goal of this project is to empower users to make smarter, more informed decisions in their fantasy leagues. The vision is to create a best-in-class tool that is both powerful for seasoned analysts and accessible to casual players. We aim to achieve this by combining sophisticated analytics with a clean, intuitive user interface.

## 1.3. Core Features

The application is built around several key features:

-   **Trade Analysis**: Evaluate the fairness and impact of multi-player, multi-team trades.
-   **Schedule Swap Analysis**: Simulate swapping schedules between teams to analyze the impact of schedule luck.
-   **Live Auction Draft Tool**: A comprehensive toolkit for live auction drafts, featuring a Smart Auction Bot for real-time nomination and bidding advice.
-   **Standings Tools**: Utilities for scraping and adjusting weekly standings from Fantrax based on games-played limits.
-   **Data Exploration**: Tools for viewing raw player statistics and enriched data with draft results.
-   **Team Analyzer**: A tool to assess the categorical strengths and weaknesses of any team in the league.

## 1.4. Target Audience

The application is designed for any fantasy basketball manager who wants to gain a deeper understanding of their league and make more strategic decisions. This includes:

-   **Hardcore Analysts**: Users who love to crunch the numbers and explore every possible angle.
-   **Casual Managers**: Users who want quick, reliable advice to improve their team without hours of research.
-   **League Commissioners**: Users who can use the tools to analyze league-wide trends and fairness.

---

# 2. Application Architecture

This section provides a high-level overview of the Fantasy Trade Analyzer's architecture, explaining how the application is structured, how it starts, and how its various components interact.

## 2.1. Application Entry Point (`Home.py`)

The main entry point for the Streamlit application is `Home.py`. This script is responsible for:

- **Page Configuration**: Setting up the browser tab title, icon, and page layout.
- **Session State Initialization**: Ensuring that all necessary keys in `st.session_state` are initialized with default values on the first run.
- **Default Data Loading**: On the first visit, it automatically finds and loads the default player and schedule CSV files from the `/data` directory.
- **UI Rendering**: It renders the main landing page, which includes a welcome message and basic instructions.

## 2.2. High-Level Structure

The application follows a modular, multi-page Streamlit architecture:

- **`Home.py` (Entry Point)**: The starting point that prepares the application environment.
- **`pages/` (Application Features)**: Contains the individual Python scripts for each of the application's main features (e.g., `1_Trade_Analysis.py`).
- **`modules/` (Reusable Components)**: Contains reusable pieces of UI and logic for each feature, promoting a DRY (Don't Repeat Yourself) approach.
- **`logic/` (Core Business Logic)**: Contains the complex algorithms and data manipulation functions that power the application's features.
- **`data/` (Data Storage)**: Contains raw CSV files for player stats and schedules.

## 2.3. Data Flow

1.  **Load**: `Home.py` or the `Downloader` feature reads data into the application.
2.  **Process**: The data loading functions (`data_loader.py`, `modules/data_preparation.py`) clean, process, and enrich the data.
3.  **Store**: The processed DataFrames are stored in `st.session_state` to be accessible across the entire application.
4.  **Analyze**: When a user navigates to a feature page, the script for that page retrieves the necessary data from the session state and passes it to the appropriate functions in the `logic/` or `modules/` directories.
5.  **Display**: The results of the analysis are then rendered to the user through various UI components, including tables, charts, and metrics.

This architecture creates a clear separation of concerns, making the codebase easier to understand, maintain, and extend.
