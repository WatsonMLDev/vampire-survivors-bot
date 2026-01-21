import os
import random
import glob
from PIL import Image, ImageOps

# --- CONFIGURATION ---
MAPS_DIR = "stage_maps"
SPRITES_DIR = "enemy_sprites"
OUTPUT_DIR = "synthetic_dataset" 
IMG_SIZE = 640
MAX_ENEMIES = 100
BUFFER_SIZE = 160 
NUM_SAMPLES = 5000 # Configure number of samples to generate
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.15
TEST_SPLIT = 0.05

for split in ['train', 'val', 'test']:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# 1. Get all sprite files and create a stable class mapping
sprite_files = sorted(glob.glob(f"{SPRITES_DIR}/*.*"))
class_names = [os.path.basename(f).split('.')[0] for f in sprite_files]

with open(f"{OUTPUT_DIR}/classes.txt", "w") as f:
    f.write("\n".join(class_names))

def generate_sample(sample_id):
    map_files = glob.glob(f"{MAPS_DIR}/*.*")
    if not map_files:
        print("No maps found in stage_maps folder!")
        return

    bg_path = random.choice(map_files)
    try:
        bg = Image.open(bg_path).convert("RGBA")
    except Exception as e:
        print(f"Failed to open {bg_path}: {e}")
        return

    # Random crop from map
    bg_w, bg_h = bg.size
    left = random.randint(0, max(0, bg_w - IMG_SIZE))
    top = random.randint(0, max(0, bg_h - IMG_SIZE))
    canvas = bg.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    # Pad if the map crop is smaller than IMG_SIZE
    if canvas.size != (IMG_SIZE, IMG_SIZE):
        canvas = canvas.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    # --- Define Central Buffer Zone ---
    buffer_x1 = (IMG_SIZE - BUFFER_SIZE) // 2
    buffer_y1 = (IMG_SIZE - BUFFER_SIZE) // 2
    buffer_x2 = buffer_x1 + BUFFER_SIZE
    buffer_y2 = buffer_y1 + BUFFER_SIZE

    # Select 1-4 random enemy types for this specific swarm
    num_types = random.randint(1, min(4, len(sprite_files)))
    selected_types = random.sample(sprite_files, num_types)

    yolo_labels = []
    num_enemies = random.randint(5, MAX_ENEMIES)

    for _ in range(num_enemies):
        sprite_path = random.choice(selected_types)
        sprite = Image.open(sprite_path).convert("RGBA")

        # --- 1. Much Smaller Scaling ---
        # Clamp max size to be much smaller (e.g., 1/6th of canvas)
        max_allowed = IMG_SIZE // 6
        if sprite.width > max_allowed or sprite.height > max_allowed:
            ratio = max_allowed / max(sprite.width, sprite.height)
            sprite = sprite.resize((int(sprite.width * ratio), int(sprite.height * ratio)), Image.Resampling.LANCZOS)

        # Scale down significantly to match game proportions
        scale = random.uniform(0.50, 0.70) 
        new_w = max(1, int(sprite.width * scale))
        new_h = max(1, int(sprite.height * scale))
        sprite = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Tiny Rotation & Flip
        sprite = sprite.rotate(random.uniform(-10, 10), expand=True, resample=Image.Resampling.BICUBIC)
        if random.random() > 0.5:
            sprite = ImageOps.mirror(sprite)

        sw, sh = sprite.size

        # --- 2. Placement with Central Buffer ---
        max_attempts = 100
        placed = False
        for _ in range(max_attempts):
            # Generate a potential position
            pos_x = random.randint(0, max(0, IMG_SIZE - sw))
            pos_y = random.randint(0, max(0, IMG_SIZE - sh))

            # Define sprite's bounding box
            sprite_x2 = pos_x + sw
            sprite_y2 = pos_y + sh

            # Check for overlap with the central buffer zone
            is_overlapping = not (sprite_x2 < buffer_x1 or
                                  pos_x > buffer_x2 or
                                  sprite_y2 < buffer_y1 or
                                  pos_y > buffer_y2)

            if not is_overlapping:
                canvas.alpha_composite(sprite, (pos_x, pos_y))
                placed = True
                break
        
        if not placed:
            continue # Skip this enemy if we couldn't place it

        # Labeling
        class_id = sprite_files.index(sprite_path)
        center_x = (pos_x + sw / 2) / IMG_SIZE
        center_y = (pos_y + sh / 2) / IMG_SIZE
        norm_w = sw / IMG_SIZE
        norm_h = sh / IMG_SIZE
        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

    # Determine Split
    split = random.choices(['train', 'val', 'test'], weights=[TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT])[0]

    # Save
    final_img = canvas.convert("RGB")
    final_img.save(f"{OUTPUT_DIR}/images/{split}/sample_{sample_id}.jpg")
    with open(f"{OUTPUT_DIR}/labels/{split}/sample_{sample_id}.txt", "w") as f:
        f.write("\n".join(yolo_labels))

# Generate samples
print(f"Starting generation of {NUM_SAMPLES} samples...")
for i in range(NUM_SAMPLES):
    if i % 50 == 0: print(f"Generated {i} samples...")
    generate_sample(i)

# Create the data.yaml for YOLO training
# Assuming we run yolo from the parent directory of OUTPUT_DIR, or pointing to this yaml file
yaml_content = f"""
path: . # Dataset root relative to this yaml file (if inside synthetic_dataset_v2) or just use relative paths below
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)

print(f"\nDone! Check the '{OUTPUT_DIR}' folder.")