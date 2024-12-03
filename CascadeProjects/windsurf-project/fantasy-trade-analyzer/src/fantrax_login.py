import pickle
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def save_fantrax_cookies():
    service = Service(ChromeDriverManager().install())

    options = Options()
    options.add_argument("--window-size=1920,1600")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36")

    print("Opening Chrome browser for Fantrax login...")
    print("Please log in when the browser opens. You have 30 seconds.")
    
    with webdriver.Chrome(service=service, options=options) as driver:
        driver.get("https://www.fantrax.com/login")
        time.sleep(30)  # Wait for user to log in
        pickle.dump(driver.get_cookies(), open("fantraxloggedin.cookie", "wb"))
        print("Cookies saved successfully!")

if __name__ == "__main__":
    save_fantrax_cookies()
