# Fantasy Basketball Trade Analyzer & Auction Draft Tool

This project is a comprehensive suite of tools for fantasy basketball managers, built with Streamlit. It provides a sophisticated Trade Analyzer and a feature-rich Live Auction Draft Tool to help users make informed decisions and dominate their leagues.

## Core Features

1.  **Trade Analyzer**: Evaluate multi-player trades by analyzing player statistics, and schedule strength.
2.  **Live Auction Draft Tool**: A powerful tool to manage a fantasy basketball auction draft in real-time, with dynamic player valuations that adapt as the draft progresses.

---

## Auction Draft Tool - In-Depth Features

The Live Auction Draft Tool is designed to give you a significant strategic advantage during your auction drafts.

### Valuation Engine

-   **Player Power Score (PPS)**: At the core of the tool is the PPS, a proprietary metric that combines a player's trend-weighted historical performance (FP/G) with their risk-adjusted availability (Avg. Games Played %) to produce a single, powerful performance score.
-   **Dynamic Adjusted Values**: Player values (`AdjValue`) are not static. They are recalculated live after every single pick, factoring in the remaining money in the league, player scarcity, and roster needs.

### Customizable Valuation Models

Tailor the valuation engine to your specific draft strategy with flexible model settings:

-   **Base Value Calculation**: Choose how a player's initial `BaseValue` is determined.
    -   `Blended (VORP + Market)`: A balanced approach combining a player's Value Over Replacement Player (VORP) with their historical auction market value.
    -   `Pure VORP`: A purely analytical approach based on VORP.
    -   `Pure Market Value`: A valuation based entirely on historical auction data.
-   **In-Draft Scarcity Model**: Choose how the `AdjValue` is modified by scarcity during the draft.
    -   `Tier Scarcity`: Applies a premium to players in tiers that are becoming depleted.
    -   `Position Scarcity`: Applies a premium to players at positions that are becoming scarce.
    -   `None`: Disables scarcity adjustments.

### Advanced Draft Management

-   **Injury Management System**: Before the draft, mark players as injured for a "Half Season" or "Out for Season". The projection engine will automatically adjust their PPS and value accordingly.
-   **Undo Last Pick**: Made a mistake? The "Undo Last Pick" button instantly reverses the last selection, returning the player to the pool and refunding the team's budget.
-   **Live Draft Summary**: A summary table tracks every pick, comparing the `DraftPrice` to the `AdjValue` at the time of the pick. It color-codes each pick as a "Bargain" (green) or a "Reach" (red), giving you instant feedback on your draft performance.

---

## Core Programming Principles

This project adheres to a strict set of programming principles to ensure high-quality, maintainable, and robust code.

-   **Clean & Readable Code**: Prioritize clarity and simplicity.
-   **Efficiency**: Use efficient algorithms and data structures.
-   **Robust Error Handling**: Implement comprehensive error handling to prevent crashes.
-   **Modularity & Reusability**: Write modular, reusable code to avoid duplication.
-   **Security**: Follow secure coding practices to protect data.
-   **Simplicity**: Keep the design and implementation as simple as possible.

## Code Style Guidelines

-   **Indentation**: Tabs
-   **Naming Conventions**: `snake_case` for variables, `PascalCase` for classes, `camelCase` for functions.
-   **Comments**: Use clear and concise comments where necessary.
-   **Line Length**: Keep lines under 100 characters.

## How to Run

1.  Ensure you have Python and `pip` installed.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

This project is a sophisticated, multi-purpose tool for fantasy basketball enthusiasts. It started as a trade analyzer and has now been expanded to include a powerful, dynamic auction draft assistant designed to give you a significant edge in your fantasy league.

## Auction Draft Tool Features

The auction draft tool is built on a multi-layered valuation model that provides dynamic, real-time insights during your live draft.

-   **Player Power Score (PPS) Engine**: At its core, the tool uses a PPS model that calculates a forward-looking, risk-adjusted score for every player. It analyzes trend-weighted performance over recent seasons and factors in games played to assess availability and risk.

-   **Advanced VORP Model**: It implements a Value Over Replacement Player (VORP) system that translates the raw PPS scores into a clear measure of each player's value relative to the other players in the pool.

-   **Dynamic Tiering System**: To reflect the premium on elite talent, the tool segments players into five tiers. It applies weights to the VORP scores based on these tiers, ensuring that top players are valued appropriately. This system is fully dynamic and re-tiers the available player pool after every pick.

-   **Live Dynamic Valuations**: A player's 'AdjValue' (Adjusted Value) recalculates in real-time after each draft pick. This value responds to both the money leaving the market and the shifting tiers of available talent, giving you a live, accurate assessment of player worth.

-   **Complete Draft Management**: The UI fully supports a live draft. You can assign players to teams, and the tool will automatically track team budgets, enforce spending limits (ensuring every team can fill their roster), and display detailed team rosters.

-   **Customizable League Settings**: The tool is highly flexible, allowing you to configure the number of teams, budget per team, roster spots, and even the number of games in a season to fine-tune the projections.

## How to Use

1.  **Launch the App**: Run the Streamlit application.
2.  **Navigate to the Tool**: Select the 'Auction Draft Tool' from the sidebar navigation.
3.  **Configure Settings**: In the sidebar, set your league's parameters (number of teams, budget, etc.) and generate the initial PPS projections.
4.  **Start the Draft**: Click 'Start Draft' to load the player pool with their initial calculated values.
5.  **Draft Players**: As players are won in your auction, use the 'Draft a Player' form to select the player, the winning team, and the price. The tool will automatically update all team budgets and recalculate the adjusted values for all remaining players.
6.  **Monitor Rosters**: Use the expandable team roster views at the bottom to track each team's progress, budget, and drafted players.

## Future Enhancement Ideas

This tool provides a powerful foundation that can be expanded even further. Here are some ideas for future development:

-   **Keeper League Support**: Add functionality to input keeper players and their associated salaries before the draft begins. This would automatically adjust team budgets and remove those players from the available pool.

-   **Improved Multi-Position Eligibility**: Enhance the VORP model to more accurately value players who are eligible for multiple positions, as they offer greater roster flexibility.

-   **Live Inflation Index**: Create a visual gauge or metric that shows the current market inflation or deflation. This would track whether players are generally being drafted for more or less than their adjusted values, indicating market trends.

-   **Post-Draft Grades**: After the draft is complete, generate a 'Draft Grade' for each team based on the total value they acquired versus the money they spent.

-   **Enhanced UI/UX**: 
    -   A classic **Draft Board View** showing all drafted players under their respective teams.
    -   **Advanced Filtering/Searching** for the available players table (e.g., by position, tier, etc.).
    -   A dedicated, more detailed **Team Page** view.

-   **Historical Draft Analysis**: Allow users to load past draft results (from a CSV, for example) to analyze trends, team strategies, and the historical accuracy of the valuation model.



### Auction Draft Tool (In Development)

We are currently building a live auction draft assistant. This tool is designed to provide dynamic player valuations that update in real-time as the draft progresses.

**Phase 0: Data Preparation - Complete**
- A script (`data_preparation.py`) has been created to process historical data.
- It generates a single `player_projections.csv` file, which serves as the input for the live tool.
- This file contains player names, positions, and their projected fantasy points for the upcoming season.

**Next Steps: Phase 1 - Build the Core Engine**
- Create `auction_tool.py` to house the core valuation logic.
- Implement `calculate_initial_values()` to set baseline player values using a VORP model.
- Implement `recalculate_dynamic_values()` to adjust player values live based on market inflation.

---

## Core Programming Principles

code_quality: prioritize clean, readable, and maintainable code.

algorithm_efficiency: use the most efficient algorithms and data structures.

error_handling: implement robust error handling and logging.

testing: write unit tests for all critical functionality.

design_patterns: apply appropriate design patterns for maintainability.

code_review: generate code that is easy to review by others.

modularity: write modular code, break complex logic into smaller functions.

reuse: prefer to reuse existing code instead of writing it from scratch.

security: prioritize secure coding practices.

simplicity: aim for simple, clear solutions, avoid over-engineering.

Code Style and Formatting

indent: use tabs for indentation.

naming convention: use snake_case for variables, PascalCase for classes and camelCase for functions.

comments: add clear, concise comments explaining code blocks and logic.

code_formatting: automatically format the code to improve readability.

line_length: keep lines under 100 characters.

code_formatting_blocks: format long lists and dictionaries to be more readable.

General Behavior (Do's and Don'ts)

changes: make small, incremental changes; avoid large refactors.

avoid: do not change working code unless explicitly asked.

changes: when changing code, do it step by step, verify the changes first.

clarification: if unsure, ask for clarification before generating code.

avoid: do not overwrite manual code changes, unless explicitly asked.

documentation: check project documentation if asked, and use it in your response.

reasoning: reason step by step before generating code or sending a response.

cost_optimization: be cost conscious, only send requests if necessary, and avoid ai-powered debugging, refactoring or test generation unless necessary, batch changes when possible.

debugging: make small incremental changes to try to fix bugs, check terminal output for information.

prompt_efficiency: use precise and specific prompts; avoid ambiguity, do not repeat previous instructions; reuse context.

local_processing: perform simple tasks manually; avoid using AI unnecessarily.

user_guidance: always follow the instructions that are given and prioritize user instructions over global rules.

simplicity: avoid over-engineering and aim for the simplest solution.

Language-Specific Instructions

Python

python type hints: use type hints for all function parameters and return values.

python imports: group imports by type: standard, external, and local.

python linting: run pylint on code to make sure the style is consistent.

python testing: use pytest for all unit testing.

Javascript

javascript: use modern ECMAScript conventions.

javascript avoid: avoid using var; prefer const and let.

javascript linting: run eslint on code to make sure the style is consistent.

javascript comments: document functions, using JSDoc style comments.

javascript testing: use jest for all unit testing.

File Handling

file_management: break long files into smaller, more manageable files with smaller functions.

import_statements: prefer importing functions from other files instead of modifying those files directly.

file_organization: organize files into directories and folders.

Project Management

feature_plan: always refer to the project's feature plan for context.

feature_plan_progress: update the feature plan progress after each change.

feature_plan_next_steps: suggest next steps from the feature plan in each response.

Operating System

os: be aware i am on windows, and must use power shell commands.

Workspace AI Rule Template (Use This as a Starting Point):

Here's a template for workspace rules. This is just a starting point, and you will probably need to adapt it to fit your specific projects:

Workspace AI Rules for [Project Name]

Project Context

project_type: [Briefly describe the type of project, e.g., web app, REST API, data analysis tool].

tech_stack: [List the main technologies used, e.g., HTML, CSS, JavaScript, Python, React].

file_structure: [Describe the recommended file and directory structure].

api_integration: [If using an API, specify requirements here].

api_endpoint: [If using an API, specify an endpoint here].

api_authentication:[If using an API, specify the type of authentication here]

api_other:[If using an API, any other specific requirements]

database: [If using a database, specify the database used here]

database_schema: [If using a database, specify the database schema here]

Feature Requirements

basic_sections: [If a web application, specify the basic sections].

form_functionality: [If a form is used, specify the requirements].

responsive_design: [Specify if the app must be responsive].

navigation: [Specify the navigation structure]

authentication: [Specify if there should be authentication]

authorization: [Specify if there should be authorization]

state_management: [Specify if any state management should be used]

payments: [Specify if there are payments being used]

other: [List any other features that are not covered].

Styling and Design

css_framework: [Specify the preferred CSS framework or style].

css_libraries:[Specify the preferred CSS libraries]

ui_library: [Specify the preferred UI library]

design_system: [Specify the preferred design system]

JavaScript Instructions

javascript_validation: [Specify any javascript validation requirements].

javascript_libraries: [Specify the javascript libraries to use].

javascript_frameworks: [Specify the javascript frameworks to use].

javascript_state_management: [Specify if state management must be done with javascript]

other: [List any other javascript specific requirements].

Testing

testing: [Specify any testing instructions for the code, both back end and front end]

testing_libraries: [Specify any libraries that should be used for testing]

testing_frameworks:[Specify any testing frameworks that must be used]

Project Management

feature_plan: always refer to the project's feature plan as a guide.

documentation: always refer to project documentation if it exists.

other: [Any other project management guidelines]

Example Workspace AI Rules (Landing Page):

Here's an example of workspace rules for a landing page project:

Project Context

project_type: landing page generator.

tech_stack: HTML, CSS, JavaScript, External API.

file_structure: organize files into directories: index.html, css/styles.css, js/scripts.js, and images in images/.

Feature Requirements

basic_sections: The landing page must include a header, hero section, features section, contact form, and footer.

form_functionality: The contact form should be able to collect name, email, and message fields, and make sure that form validation is done via javascript.

api_endpoint: The API endpoint must be /submit, with a redirect to /thankyou after a successful submission.

responsive_design: The page must be responsive on different screen sizes.

navigation: The header must include links to the correct sections of the page.

Styling and Design

css_framework: use basic CSS, avoid frameworks.

JavaScript Instructions

javascript_validation: implement form validation via javascript.

Project Management

feature_plan: use the provided feature plan as a guide.





Development Best Practices
Component Dependencies
When building interconnected systems:

Document all dependencies between components explicitly

Changes to one component must cascade appropriately to all dependent components

Maintain a “single source of truth” for shared configurations

Example: If component A defines data structure, components B and C must adapt to match

Systematic Change Protocol
Before making ANY changes:

Identify Impact Area

Which components will be affected?

What dependencies exist?

Is this UI-only or does it affect core logic?

Document Current State

What’s currently working?

What’s the current error/issue?

Which files will be affected?

Plan Single Change

One file at a time

One feature at a time

One bug at a time

Verify Testing Capability

Can we test this change immediately?

Do we have all dependencies?

What specific steps will verify success?

If not testable, what’s missing?

Development Philosophy
Preserve What Works

Don’t modify working components without necessity

Keep established patterns

Maintain proven workflows

Simplify Everything

Remove unnecessary abstraction layers

Simplify class hierarchies

Keep UI components focused

Avoid over-engineering

Change Carefully

Verify before changing

One change at a time

Test each change

No premature optimization

Stay Focused

Complete one step before starting another

Don’t get sidetracked by improvements

Keep systematic sequence

Document everything

Test Real Usage First

Never implement without ability to test

Verify all dependencies exist

Must be able to manually verify

If you can’t test it, don’t build it

Version Control Strategy
Maintain multiple source versions for stability

Keep last known good version

Create dedicated test versions

Label versions clearly with date and purpose