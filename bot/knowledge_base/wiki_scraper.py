import requests
from bs4 import BeautifulSoup
import json

def scrape_evolutions():
    url = "https://vampire-survivors.fandom.com/wiki/Evolution"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch page: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Tables often contain the evolution data
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables.")
    
    evolutions = []

    for i, table in enumerate(tables):
        rows = table.find_all('tr')
        if not rows:
            continue
            
        print(f"Table {i} has {len(rows)} rows.")
        
        # Heuristic: Look for headers like "Base Weapon", "Passive Item", "Evolved Weapon"
        headers = [th.get_text(strip=True).lower() for th in rows[0].find_all('th')]
        print(f"Table {i} Headers: {headers}")
        
        # Try to extract data if it looks like an evolution table
        if any("base weapon" in h for h in headers):
            for row in rows[1:]:
                cols = row.find_all(['td'])
                # The table likely has image columns, so text content might be sparse or in specific indices
                # Based on header: ['', 'Base weapon', '', 'Passive item', '', 'Evolution']
                # Indices: 0=Icon, 1=Name, 2=Icon, 3=Name, 4=Icon, 5=Name
                
                if len(cols) >= 6:
                    base_weapon = cols[1].get_text(strip=True)
                    passive_item = cols[3].get_text(strip=True)
                    evolution = cols[5].get_text(strip=True)
                    
                    if base_weapon and evolution:
                        evolutions.append({
                            "base_weapon": base_weapon,
                            "passive_item": passive_item,
                            "evolution": evolution
                        })

    with open('bot/knowledge_base/evolutions.json', 'w') as f:
        json.dump(evolutions, f, indent=2)
    
    print(f"Dumped {len(evolutions)} evolution rows to bot/knowledge_base/evolutions.json")

if __name__ == "__main__":
    scrape_evolutions()
