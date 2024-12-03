import logging
import pickle
from fantraxapi.fantrax import FantraxAPI
from requests import Session

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load cookies from the file
with open('fantraxloggedin.cookie', 'rb') as cookie_file:
    cookies = pickle.load(cookie_file)

# Example usage of FantraxAPI
class FantraxAnalysisTab:
    def __init__(self, league_id: str, session=None):
        if session is None:
            session = Session()
            session.cookies.clear()  # Clear any existing cookies
            for cookie in cookies:
                session.cookies.set(cookie['name'], cookie['value'], domain=cookie.get('domain'))
        self.api = FantraxAPI(league_id, session)

    def analyze_standings(self):
        standings = self.api.standings()
        # Perform analysis on standings
        logging.info("Standings Analysis:")
        for team in standings.teams:
            logging.info(f"Team: {team.name}, Wins: {team.wins}, Losses: {team.losses}")

    def analyze_trade_block(self):
        trade_blocks = self.api.trade_block()
        # Perform analysis on trade blocks
        logging.info("Trade Block Analysis:")
        for block in trade_blocks:
            logging.info(f"Team: {block.team.name}, Players: {[player.name for player in block.players]}")

# Example function to run analysis
if __name__ == "__main__":
    league_id = "6zeydg0cm03y4myx"
    analysis_tab = FantraxAnalysisTab(league_id)
    analysis_tab.analyze_standings()
    analysis_tab.analyze_trade_block()
