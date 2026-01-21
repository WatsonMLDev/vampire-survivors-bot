import os
import requests
import time

def get_wiki_images(api_url, category=None, pages=None):
    """
    Fetches image URLs from a wiki API.
    :param api_url: The API endpoint (e.g., https://site.fandom.com/api.php)
    :param category: Optional category name to fetch from (e.g., 'Category:Enemy_sprites')
    :param pages: Optional list of page titles to fetch images from (e.g., ['Mad Forest'])
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "ImageDownloader/1.0 (Educational Project)"})
    
    file_titles = []

    # 1. Get file titles from a Category
    if category:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file",
            "cmlimit": "max",
            "format": "json"
        }
        res = session.get(api_url, params=params).json()
        file_titles.extend([m['title'] for m in res.get('query', {}).get('categorymembers', [])])

    # 2. Get file titles from specific Pages (for Stage galleries)
    if pages:
        params = {
            "action": "query",
            "titles": "|".join(pages),
            "prop": "images",
            "imlimit": "max",
            "format": "json"
        }
        res = session.get(api_url, params=params).json()
        query_pages = res.get('query', {}).get('pages', {})
        for pid in query_pages:
            if 'images' in query_pages[pid]:
                file_titles.extend([img['title'] for img in query_pages[pid]['images']])

    # 3. Resolve actual URLs for all collected file titles
    image_urls = {}
    for i in range(0, len(file_titles), 50):  # API allows batches of 50
        batch = file_titles[i:i+50]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json"
        }
        res = session.get(api_url, params=params).json()
        info_pages = res.get('query', {}).get('pages', {})
        for pid in info_pages:
            title = info_pages[pid].get('title')
            if 'imageinfo' in info_pages[pid]:
                image_urls[title] = info_pages[pid]['imageinfo'][0]['url']
                
    return image_urls

def download_files(url_dict, folder_name, filter_keywords=None):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for title, url in url_dict.items():
        # Apply optional filter (e.g., only download files with 'Map' in the name)
        if filter_keywords and not any(k.lower() in title.lower() for k in filter_keywords):
            continue

        # Clean filename and strip Fandom revision suffixes
        filename = title.replace("File:", "").replace(" ", "_").replace(":", "_")
        clean_url = url.split("/revision/")[0] if "fandom.com" in url else url
        
        filepath = os.path.join(folder_name, filename)
        if os.path.exists(filepath):
            continue

        print(f"Downloading: {filename}")
        try:
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
            time.sleep(0.2)  # Respect the server
        except Exception as e:
            print(f"Failed {title}: {e}")

# --- EXECUTION ---

# 1. Download Enemy Sprites from Fandom
# fandom_api = "https://vampire-survivors.fandom.com/api.php"
# print("Fetching enemy sprites...")
# sprites = get_wiki_images(fandom_api, category="Category:Enemy_sprites")
# download_files(sprites, "enemy_sprites")

# 2. Download Stage Maps from the main wiki
# Adding several stages to the list
# API endpoint for the main wiki
stage_wiki_api = "https://vampire.survivors.wiki/api.php"

# We target the 'Stages' page directly because it contains the tables
stage_list_page = ["Stages"]

print("Fetching images from the Stages summary page...")
# This retrieves every image linked on https://vampire.survivors.wiki/w/Stages
all_stage_images = get_wiki_images(stage_wiki_api, pages=stage_list_page)

# Filter specifically for 'Preview' to capture the "Preview of normal..." images
# and ignore UI icons like gold, items, or character heads.
download_files(all_stage_images, "stage_previews", filter_keywords=["Preview"])