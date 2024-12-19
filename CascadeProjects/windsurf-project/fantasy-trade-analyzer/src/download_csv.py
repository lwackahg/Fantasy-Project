from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
from datetime import datetime, timedelta
import os
import shutil

def login_to_fantrax(driver):
    """
    Log in to Fantrax website using the login form.
    """
    try:
        print("Navigating to Fantrax homepage...")
        driver.get("https://www.fantrax.com/login")
        time.sleep(5)  # Wait for page to load
        
        print("Attempting to find login form...")
        
        # Try different methods to find login fields
        try:
            # First attempt - direct login form
            email_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email']"))
            )
            password_input = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
        except:
            print("Direct login form not found, trying iframe...")
            # Try to switch to iframe if present
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            if iframes:
                for iframe in iframes:
                    try:
                        driver.switch_to.frame(iframe)
                        email_input = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email']"))
                        )
                        password_input = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
                        break
                    except:
                        driver.switch_to.default_content()
                        continue
            else:
                raise Exception("No login form found")

        print("Found login form fields, entering credentials...")

        # Clear and enter email
        email_input.clear()
        time.sleep(1)
        email_input.send_keys("gabrys8@hotmail.com")
        time.sleep(1)

        # Clear and enter password
        password_input.clear()
        time.sleep(1)
        password_input.send_keys("lukas1911")
        time.sleep(1)

        # Try to find and click login button
        try:
            login_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
            )
            print("Found login button, clicking...")
            login_button.click()
        except:
            print("Login button not found, trying form submit...")
            password_input.submit()

        # Wait for login to complete
        print("Waiting for login to complete...")
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".league-name, .league-header, .user-menu"))
        )
        print("Successfully logged in")
        return True

    except Exception as e:
        print(f"Login failed with error: {str(e)}")
        try:
            driver.save_screenshot("login_error.png")
            print("Screenshot saved as login_error.png")
            print(f"Current URL: {driver.current_url}")
            with open("page_source.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("Page source saved to page_source.html")
        except:
            pass
        return False

def download_fantrax_csv(days_back=30, webdriver_path=None, output_path=None):
    """
    Download player stats CSV from Fantrax for a specified date range.
    """
    driver = None
    try:
        # Set up Chrome options for download handling
        chrome_options = Options()
        
        # Add options to handle SSL certificate issues
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ignore-ssl-errors')
        chrome_options.add_argument('--allow-insecure-localhost')
        
        # Add options to improve stability
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-software-rasterizer')
        
        if output_path:
            download_dir = os.path.dirname(os.path.abspath(output_path))
            chrome_options.add_experimental_option('prefs', {
                'download.default_directory': download_dir,
                'download.prompt_for_download': False,
                'download.directory_upgrade': True,
                'safebrowsing.enabled': True,
                'profile.default_content_setting_values.automatic_downloads': 1
            })

        # Set up the Selenium WebDriver
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)

        # Login to Fantrax
        if not login_to_fantrax(driver):
            raise Exception("Failed to login to Fantrax")

        print("Calculating date range...")
        # Calculate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Construct URL with dynamic dates
        url = f"https://www.fantrax.com/fantasy/league/6zeydg0cm03y4myx/players;reload=1;statusOrTeamFilter=ALL_TAKEN;pageNumber=1;startDate={start_date_str};endDate={end_date_str};timeframeTypeCode=BY_DATE"
        
        print(f"Navigating to players page: {url}")
        driver.get(url)
        time.sleep(10)

        print("Looking for download button...")
        try:
            download_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Download all as CSV')]"))
            )
            print("Found download button, attempting to click...")
            driver.execute_script("arguments[0].click();", download_button)
            print("Clicked download button")
        except Exception as e:
            print(f"Failed to click download button: {str(e)}")
            driver.save_screenshot("download_error.png")
            print("Screenshot saved as download_error.png")
            print(f"Current URL: {driver.current_url}")
            with open("download_page_source.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("Page source saved to download_page_source.html")
            raise
        
        print("Waiting for download to complete...")
        time.sleep(15)

        if output_path:
            print(f"Looking for downloaded file in: {download_dir}")
            files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith('.csv')]
            if files:
                latest_file = max(files, key=os.path.getctime)
                shutil.move(latest_file, output_path)
                print(f"CSV file saved to: {output_path}")
            else:
                print("No CSV files found in download directory")
                raise Exception("No CSV file found in download directory")
        
    except Exception as e:
        print(f"Error downloading CSV: {str(e)}")
        raise
        
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    output_path = os.path.join(data_dir, "Fantrax-Players-30.csv")
    download_fantrax_csv(days_back=30, output_path=output_path)
