# Fantasy Trade Analyzer & Auction Draft Tool

This project provides tools for fantasy basketball analysis, including a trade analyzer and a new live auction draft assistant.

## Project Status

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