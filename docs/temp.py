import pandas as pd 
import requests
from bs4 import BeautifulSoup
import csv
import time

# Load the CSV file
df = pd.read_csv("docs/PeerGPT - sample RCTs list - Sheet1.csv")  # use forward slash for cross-platform paths

# Clean and prepare URL list
rctsLinks = df['URL'].dropna().tolist()  # Remove NaNs and convert to list
cleaned_urls = []

for url in rctsLinks:
    if not str(url).startswith("http"):
        url = "https://" + str(url).strip()
    cleaned_urls.append(url)

# Print sample to verify
print("✅ Sample URLs:")
print(cleaned_urls[:5])

# Output CSV file
output_file = "pubmed_abstracts.csv"

# Open the CSV and write headers
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["URL", "Abstract"])  # Header row

    for url in cleaned_urls:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract abstract
            abstract_div = soup.find("div", class_="abstract-content selected")
            abstract = abstract_div.get_text(strip=True) if abstract_div else "Abstract not found"

            writer.writerow([url, abstract])
            print(f"✅ Extracted: {url}")

            time.sleep(1)  # Polite delay

        except Exception as e:
            print(f"❌ Failed: {url} - {e}")
            writer.writerow([url, "Error during extraction"])
