# Fantasy Trade Analyzer: Technical Documentation

Welcome to the comprehensive technical documentation for the Fantasy Trade Analyzer. This collection of documents provides a complete overview of the project's architecture, development practices, and core features.

---

## Part 1: Project Foundations

This section covers the high-level principles and structure of the project.

-   [**010: Project Overview**](./010_Project_Overview.md)
    -   *An introduction to the project's purpose, vision, core features, and target audience.*
-   [**020: Core Principles & Style Guide**](./020_Core_Principles_and_Style_Guide.md)
    -   *The foundational programming principles and code style guidelines for all contributors.*
-   [**030: System Architecture & Data Flow**](./030_System_Architecture_and_Data_Flow.md)
    -   *A detailed breakdown of the application's architecture, data handling pipelines, and core logic modules.* Includes a **Team Mapping Resolution** subsection documenting centralized team mappings and seasonal update steps.
-   [**035: Data Dictionary**](./035_Data_Dictionary.md)
    -   *A guide to the primary data sources, explaining the purpose and schema of key CSV files and DataFrames.*
-   [**040: Development Workflow**](./040_Development_Workflow.md)
    -   *Guidelines for the development process, including branching, testing, and code review.*
-   [**050: Future Ideas & Roadmap**](./050_Future_Ideas_and_Roadmap.md)
    -   *A living document outlining potential new features and long-term goals.*

---

## Part 2: Feature Deep Dives

This section provides detailed, technical explanations for each of the application's major features.

-   [**130: Team Analyzer**](./130_Feature_Deep_Dive_-_Team_Analyzer.md)
    -   *Assessing team strengths and weaknesses with statistical analysis and radar charts.*
-   [**140: Fantrax Downloader**](./140_Feature_Deep_Dive_-_Downloader.md)
    -   *Automating data fetching from Fantrax using Selenium and Requests.*
-   [**150: Schedule Analysis**](./150_Feature_Deep_Dive_-_Schedule_Analysis.md)
    -   *Simulating schedule swaps to analyze the impact of luck on league standings.*
-   [**160: Auction Draft Tool**](./160_Feature_Deep_Dive_-_Auction_Draft_Tool.md)
    -   *A real-time assistant for live auction drafts with a valuation engine and AI bot.*
-   [**170: Player Full Data Explorer**](./170_Feature_Deep_Dive_-_Player_Full_Data.md)
    -   *Exploring and filtering raw player statistics with integrated draft results.*
-   [**180: Trade Analysis**](./180_Feature_Deep_Dive_-_Trade_Analysis.md)
    -   *Evaluating multi-player, multi-team trades with detailed impact analysis.*
-   [**190: Standings Tools**](./190_Feature_Deep_Dive_-_Standings_Tools.md)
    -   *Scraping and applying weekly standings adjustments for commissioners.*
-   [**200: Player Game Log Scraper**](./200_Feature_Deep_Dive_-_Player_Game_Log_Scraper.md)
    -   *Scraping game-by-game player stats with variability analysis and consistency metrics.*

---

## Full File Index

Below is a complete list of all Python files in the project, categorized by their function, for quick reference.

### Core Application
- `Home.py`
- `__init__.py`
- `config.py`
- `session_manager.py`
- `debug.py`

### Data Handling
- `data_loader.py`
- `fantrax_downloader.py`
- `modules/data_preparation.py`

### Logic Layer
- `logic/auction_tool.py`
- `logic/optimizer_strategies.py`
- `logic/schedule_analysis.py`
- `logic/smart_auction_bot.py`
- `logic/team_optimizer.py`
- `logic/ui_components.py`

### UI Layer (Pages)
- `pages/1_Trade_Analysis.py`
- `pages/2_Player_Consistency.py` (Public viewer for cached consistency data)
- `pages/3_Schedule_Analysis.py`
- `pages/4_Player_Full_Data.py`
- `pages/5_Team_Analyzer.py`
- `pages/6_Admin_Tools.py` (Password-protected: Downloader, Player Game Logs Scraper, Weekly Standings, Standings Adjuster)
- `pages/7_Auction_Draft_Tool.py`

### Modules
- `modules/auth/ui.py`
- `modules/data_loader_ui/ui.py`
- `modules/fantrax_downloader/logic.py`
- `modules/fantrax_downloader/ui.py`
- `modules/player_data/logic.py`
- `modules/player_data/ui.py`
- `modules/sidebar/ui.py`
- `modules/standings_adjuster/logic.py`
- `modules/standings_adjuster/ui.py`
- `modules/team_analyzer/logic.py`
- `modules/team_analyzer/ui.py`
- `modules/trade_analysis/logic.py`
- `modules/trade_analysis/ui.py`
- `modules/weekly_standings_analyzer/logic.py`
- `modules/weekly_standings_analyzer/ui.py`
- `modules/player_game_log_scraper/logic.py`
- `modules/player_game_log_scraper/ui.py`
- `modules/team_mappings.py`

### Tests
- `tests/test_app.py`
- `tests/test_config.py`
- `tests/test_data_loader.py`
- `tests/test_debug.py`
- `tests/test_player_data_display.py`
- `...and more`
