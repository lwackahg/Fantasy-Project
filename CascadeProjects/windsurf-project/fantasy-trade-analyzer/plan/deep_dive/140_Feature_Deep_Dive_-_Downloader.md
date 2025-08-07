# Deep Dive: Fantrax Data Downloader

This document details the Fantrax Data Downloader, a critical feature for keeping the application's data up-to-date.

---

### 1. Purpose

The downloader provides a user-friendly interface within the Streamlit application to fetch the latest player statistics directly from Fantrax. It automates the process of logging in, navigating to the correct reports, and downloading CSV files for multiple time ranges.

---

### 2. Architecture

The feature is organized into three main components:

- **Page Entry Point (`pages/3_Downloader.py`)**: The Streamlit page that users navigate to. It sets the page title and calls the UI module.

- **UI Module (`modules/fantrax_downloader/ui.py`)**: This module renders the user interface, which includes:
  - A dropdown to select the target league (populated from environment variables).
  - A primary button to initiate the download process for all standard time ranges (`YTD`, `60 days`, `30 days`, `14 days`, `7 days`).
  - A progress bar and status text to provide real-time feedback during the download.
  - An expandable log viewer to display the success or failure messages for each downloaded file.

- **Logic Module (`modules/fantrax_downloader/logic.py`)**: This is the engine of the feature. It uses Selenium and `requests` to perform the download operations.

---

### 3. Configuration

For the downloader to function, a `fantrax.env` file must be present in the project's root directory with the following variables:

- `FANTRAX_USERNAME`: Your Fantrax account username.
- `FANTRAX_PASSWORD`: Your Fantrax account password.
- `FANTRAX_LEAGUE_IDS`: A comma-separated list of your Fantrax league IDs.
- `FANTRAX_LEAGUE_NAMES`: A comma-separated list of names for your leagues, corresponding to the order in `FANTRAX_LEAGUE_IDS`.
- `FANTRAX_DOWNLOAD_DIR`: The local directory path where the downloaded CSV files will be saved.

---

### 4. Core Logic Breakdown

The download process is executed in the `logic.py` module and follows these steps:

1.  **Initialize Headless Browser**: A headless Google Chrome instance is started using `selenium.webdriver`. The browser is configured to automatically download files to the specified `FANTRAX_DOWNLOAD_DIR` without user prompts.

2.  **Login to Fantrax**: The script navigates to the Fantrax login page, enters the credentials from the `.env` file, and submits the form. It includes a check to ensure the login was successful.

3.  **Iterate Through Time Ranges**: The `download_all_ranges` function defines a dictionary of standard time periods. It then loops through each one to perform a download.

4.  **Construct Download URL**: For each time range, the `download_players_csv` function constructs a specific URL. This URL contains query parameters that tell the Fantrax server exactly which report to generate (e.g., league ID, date range, player status).

5.  **Generate Filename**: A descriptive filename is created based on the league name and the time range (e.g., `Fantrax-Players-My_League-(30).csv`).

6.  **Download with `requests`**: Instead of clicking a download button with Selenium (which can be unreliable), the script gets the session cookies from the logged-in Selenium browser. It then creates a `requests.Session` and adds the cookies to it. This session is used to send a direct GET request to the constructed URL, downloading the file content efficiently.

7.  **Save File**: The downloaded content is written to a CSV file in the designated download directory.

8.  **Provide Feedback**: Throughout the process, a `progress_callback` function is used to update the Streamlit UI's progress bar and status message.

9.  **Cleanup**: After all downloads are complete (or if an error occurs), the Selenium browser instance is closed to free up system resources.
