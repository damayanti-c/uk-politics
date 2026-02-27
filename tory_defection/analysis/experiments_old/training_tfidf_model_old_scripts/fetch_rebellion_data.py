"""
Fetch MP voting rebellion data from Public Whip.

This script scrapes rebellion rates from the Public Whip website.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from pathlib import Path

def fetch_rebellion_data() -> pd.DataFrame:
    """
    Scrape rebellion data from Public Whip MPs page.

    Returns:
        DataFrame with columns: mp_name, rebellion_rate
    """
    url = "https://www.publicwhip.org.uk/mps.php?sort=rebellions"

    print(f"Fetching rebellion data from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table with MP data
        table = soup.find('table')
        if not table:
            print("Could not find MP table on page")
            return pd.DataFrame()

        rows = table.find_all('tr')

        data = []
        for row in rows[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) >= 2:
                # Extract MP name (usually in first column with a link)
                name_col = cols[0].find('a')
                if name_col:
                    mp_name = name_col.text.strip()
                else:
                    mp_name = cols[0].text.strip()

                # Extract rebellion rate (look for percentage)
                rebellion_text = None
                for col in cols:
                    text = col.text.strip()
                    if '%' in text:
                        rebellion_text = text
                        break

                if rebellion_text and rebellion_text != 'n/a':
                    try:
                        rebellion_rate = float(rebellion_text.replace('%', '').strip())
                        data.append({
                            'mp_name': mp_name,
                            'rebellion_rate': rebellion_rate
                        })
                    except ValueError:
                        continue

        df = pd.DataFrame(data)
        print(f"Successfully extracted {len(df)} MPs with rebellion data")

        return df

    except Exception as e:
        print(f"Error fetching rebellion data: {e}")
        return pd.DataFrame()


def save_rebellion_data():
    """Fetch and save rebellion data to CSV."""
    df = fetch_rebellion_data()

    if not df.empty:
        output_path = Path(__file__).parent / 'source_data' / 'rebellion_rates.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"\nSaved rebellion data to: {output_path}")
        print(f"\nTop 10 rebels:")
        print(df.nlargest(10, 'rebellion_rate')[['mp_name', 'rebellion_rate']])

        return df
    else:
        print("No rebellion data retrieved")
        return None


if __name__ == '__main__':
    save_rebellion_data()
