# FantraxAPI Findings

## Useful Methods and Properties

### FantraxAPI Class

- **Initialization**: The `FantraxAPI` class is initialized with a `league_id` and an optional `Session` object.
- **Teams Property**: Retrieves a list of teams in the league using the `getFantasyTeams` endpoint.
- **Positions Property**: Likely retrieves information about player positions.
- **Trade Block Method**: Retrieves trade block information, useful for understanding available trades and player availability.
- **Transactions Method**: Fetches a list of transactions, which can be analyzed for player movement and team changes.
- **Max Goalie Games This Week Method**: Provides the maximum number of games a goalie can play this week, useful for lineup decisions.
- **Playoffs Method**: Retrieves playoff periods and matchups, essential for playoff analysis and strategy.
- **Roster Info Method**: Fetches roster information for a specific team, valuable for team management and analysis.

## Additional Methods

- **Scoring Periods Method**: Retrieves `ScoringPeriod` objects for the league, which could be useful for analyzing game schedules and results.
- **Standings Method**: Provides `Standings` objects for the current moment or a specific week, allowing for analysis of team performance over time.
- **Pending Trades Method**: Retrieves a list of pending trades, which can be useful for trade analysis and decision-making.

## Next Steps
- Continue exploring the `FantraxAPI` class for methods related to player statistics or game history.
