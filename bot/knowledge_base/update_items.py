import requests
from bs4 import BeautifulSoup
import json
import os
import re

WEAPONS_URL = "https://vampire.survivors.wiki/w/Weapons"
PASSIVES_URL = "https://vampire.survivors.wiki/w/Passive_items"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "items.json")

def scrape_tables(url, category, existing_db):
    print(f"Scraping {category} from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Determine items to skip or normalize
        # Skipping 'Evolution' entries if they are just describing the process, 
        # but picking up actual evolved weapon names is good.
        
        # Strategies:
        # 1. Look for tables with class "wikitable"
        # 2. Iterate rows, look for the first link in the first cell (usually the icon + name)
        
        tables = soup.find_all("table", class_="wikitable")
        print(f"Found {len(tables)} tables.")
        
        count = 0
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all(["th", "td"])
                if not cols:
                    continue
                
                # Usually Item Name is in the first or second column. 
                # Let's look for an <a> tag with a title attribute.
                
                found_item = False
                for col in cols[:2]: # Check first two columns
                    link = col.find("a")
                    if link and link.get("title"):
                        name = link.get("title")
                        
                        # Clean up name (remove " (Weapon)", etc if present)
                        name = re.split(r' \((Weapon|Passive)\)', name)[0]
                        name = name.strip()
                        
                        # Filter out garbage
                        if name in ["Evolution", "Union", "Level up", "Treasure Chest", "Gem", "Coin", "Floor Chicken", "Vacuum", "Rosary", "Nduja Fritta Tanto", "Orologion", "Gold Finger"]:
                           # Some of these are pickups, not inventory items (passives/weapons)
                           # Rosary etc are pickups.
                           # But "Vacuum" is listed as pickup.
                           # We strictly want things that go in the inventory.
                           if category == "weapon" and name in ["Vacuum", "Rosary", "Orologion", "Nduja Fritta Tanto", "Little Clover"]:
                               continue
                        
                        # Add to DB
                        # If conflict, prioritize weapon (some passives might share names? Unlikely)
                        if name not in existing_db:
                            existing_db[name] = category
                            count += 1
                            found_item = True
                            # print(f"  Found {category}: {name}")
                            break # Found the item in this row
        
        print(f"Added {count} new {category}s.")

    except Exception as e:
        print(f"Error scraping {url}: {e}")

def manual_overrides(db):
    # Add items that might be tricky to scrape or are new
    overrides = {
        "Spinach": "passive",
        "Armor": "passive",
        "Hollow Heart": "passive",
        "Pummarola": "passive",
        "Empty Tome": "passive",
        "Candelabrador": "passive",
        "Bracer": "passive",
        "Spellbinder": "passive",
        "Duplicator": "passive",
        "Wings": "passive",
        "Attractorb": "passive",
        "Clover": "passive",
        "Crown": "passive",
        "Stone Mask": "passive",
        "Skull O'Maniac": "passive",
        "Torrona's Box": "passive",
        "Silver Ring": "passive",
        "Gold Ring": "passive",
        "Metaglio Left": "passive",
        "Metaglio Right": "passive",
        "Runetracer": "weapon",
        "NO FUTURE": "weapon"
    }
    
    for k, v in overrides.items():
        db[k] = v

def main():
    item_db = {}
    
    scrape_tables(WEAPONS_URL, "weapon", item_db)
    scrape_tables(PASSIVES_URL, "passive", item_db)
    
    manual_overrides(item_db)
    
    # Sort and save
    sorted_db = dict(sorted(item_db.items()))
    
    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        json.dump(sorted_db, f, indent=2)
        
    print(f"Successfully saved {len(sorted_db)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
