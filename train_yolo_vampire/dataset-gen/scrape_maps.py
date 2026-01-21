import os
import requests
import time
import re

# List from your previous input
stage_names = [
    "Mad Forest", "Inlaid Library", "Dairy Plant", "Gallo Tower", "Cappella Magna", 
    "Il Molise", "Moongolow", "Holy Forbidden", "Green Acres", "The Bone Zone", 
    "Boss Rash", "Whiteout", "The Coop", "Space 54", "Carlo Cart", "Laborratory", 
    "Westwoods", "Bat Country", "Astral Stair", "Mazerella", "Tiny Bridge", 
    "Eudaimonia Machine", "Room 1665"
]

def sanitize_filename(name):
    """Removes query strings and invalid characters for Windows filenames."""
    # Remove everything from '?' onwards
    name = name.split('?')[0]
    # Remove invalid Windows chars: \ / : * ? " < > |
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def download_stage_assets(names):
    api_url = "https://vampire.survivors.wiki/api.php"
    session = requests.Session()
    
    if not os.path.exists("stage_maps"):
        os.makedirs("stage_maps")

    for stage in names:
        print(f"Processing Stage: {stage}")
        
        # 1. Get all images linked on the specific stage page
        params = {
            "action": "query",
            "titles": stage,
            "prop": "images",
            "imlimit": "max",
            "format": "json"
        }
        
        res = session.get(api_url, params=params).json()
        pages = res.get('query', {}).get('pages', {})
        
        file_titles = []
        for pid in pages:
            if 'images' in pages[pid]:
                # Filter for 'preview' or 'Tilemap' as seen in the Gallery source
                file_titles.extend([
                    img['title'] for img in pages[pid]['images'] 
                    if "preview" in img['title'].lower() or "tilemap" in img['title'].lower()
                ])

        if not file_titles:
            print(f"  [!] No preview/tilemap found for {stage}")
            continue

        # 2. Get the actual URLs for these specific files
        for i in range(0, len(file_titles), 50):
            batch = file_titles[i:i+50]
            info_params = {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json"
            }
            
            info_res = session.get(api_url, params=info_params).json()
            info_pages = info_res.get('query', {}).get('pages', {})
            
            for pid in info_pages:
                if 'imageinfo' in info_pages[pid]:
                    raw_url = info_pages[pid]['imageinfo'][0]['url']
                    original_title = info_pages[pid]['title'].replace("File:", "")
                    
                    # Sanitize the filename to prevent [Errno 22]
                    clean_filename = sanitize_filename(original_title)
                    
                    # Prevent redownloading
                    filepath = os.path.join("stage_maps", clean_filename)
                    if os.path.exists(filepath):
                        continue

                    print(f"  -> Downloading: {clean_filename}")
                    try:
                        img_data = session.get(raw_url).content
                        with open(filepath, 'wb') as f:
                            f.write(img_data)
                    except Exception as e:
                        print(f"  [X] Failed to save {clean_filename}: {e}")
        
        time.sleep(0.3)

if __name__ == "__main__":
    download_stage_assets(stage_names)