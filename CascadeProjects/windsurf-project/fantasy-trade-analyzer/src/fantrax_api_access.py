import pickle
from fantraxapi import FantraxAPI
from requests import Session

def access_fantrax_api():
    try:
        session = Session()

        print("Loading saved cookies...")
        with open("fantraxloggedin.cookie", "rb") as f:
            for cookie in pickle.load(f):
                session.cookies.set(cookie["name"], cookie["value"])

        league_id = "6zeydg0cm03y4myx"  # Updated league ID
        print(f"Connecting to league {league_id}...")

        api = FantraxAPI(league_id, session=session)
        
        # Test the connection by accessing the trade block
        print("Accessing trade block...")
        trade_block = api.trade_block()
        print("Trade Block Data:", trade_block)
        
        return api

    except FileNotFoundError:
        print("Error: Cookie file not found. Please run fantrax_login.py first to save your login session.")
    except Exception as e:
        print(f"Error accessing Fantrax API: {str(e)}")

if __name__ == "__main__":
    access_fantrax_api()
