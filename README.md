# Vampire Survivors Bot

![Showcase:](./assets/gif_bot.gif "Bot Showcase")

An advanced autonomous agent that plays Vampire Survivors using computer vision, Deep Learning, and potential field navigation. This bot combines custom-trained YOLO models with LLM capabilities to navigate the chaos, collect gems, and make strategic upgrade decisions.

## Key Features

- **Dual-Model Object Detection**:
  - **Enemies**: Powered by a YOLOv26 model trained on a hybrid dataset of real gameplay and synthetic imagery for high-fidelity enemy tracking.
  - **Items & Gems**: Utilizes the legacy YOLOv8 model (from the original fork) for gem detection.
- **Vector Field Navigation**: Implements a "Vector Pilot" system that treats enemies as repulsive forces and gems/runes as attractive forces, creating organic, fluid movement that weaves through swarms.
- **LLM Decision Making**: Integrates with LLMs (via Gemini or LiteLLM) to intelligently select upgrades during Level Up screens based on current inventory and build potential.
- **Precise UI Perception**: Uses template matching to instantly recognize game states like Level Up, Treasure Chests, and Pause screens.
- **Virtual Controller**: Emulates an Xbox 360 controller using `vgamepad` for smooth analog movement, superior to 8-direction keyboard inputs.

## Prerequisites

- **OS**: Windows (Required for `vgamepad` and `dxcam` efficiency).
- **Python**: 3.10 or higher.
- **GPU**: CUDA-capable NVIDIA GPU is highly recommended for real-time inference.
- **Drivers**: [ViGEmBus](https://github.com/ViGEm/ViGEmBus/releases) must be installed for virtual controller support.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/vampire-survivors-bot.git
   cd vampire-survivors-bot
   ```

2. **Set up a Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install PyTorch (Manual Step)**
   You must install a version of PyTorch that matches your CUDA version. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to generate the correct command.
   *Example:*
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Model Training & Datasets

Code related to training the custom YOLO model can be found in the `train_yolo_vampire/` directory.

### Datasets used for "YoloV26"
- **Legacy (Base)**: `train_yolo_vampire/dataset/synthetic_dataset.7z`
  - The first training pass. A fully synthetic dataset covering 300+ classes.
- **Current (Refined)**: `train_yolo_vampire/dataset/synthetic_dataset_real/`
  - The active dataset for the current model. It contains a curated mix of synthetic images and real gameplay captures to improve generalization.

### Directory Structure (`train_yolo_vampire/`)
- `train.py` / `finetune.py`: Entry points for training new models.
- `dataset-gen/`: Scripts used to generate the synthetic imagery.
- `results/`: Stores training metrics, confusion matrices, and model checkpoints after training runs.

[**Download Dataset**](#) *(Placeholder Link)*

## Data Recording (NitroGen)

This bot is designed to serve as a high-fidelity data collection agent for training the NVIDIA NitroGen model. It records four synchronized streams of data found in the `training_data/` directory:

1.  **Gameplay Video** (`capture_*.mp4`):
    - A clean, 60 FPS recording of the game window.
    - Used for training visual encoders.
2.  **Controller Inputs** (`capture_*.jsonl`):
    - Precise frame-by-frame logging of the virtual Xbox 360 controller states (Joystick axes, button presses).
    - Used for behavioral cloning and action prediction.
3.  **LLM Decisions** (`decisions_*.jsonl`):
    - Logs of every Level-Up decision made by the LLM.
    - Includes the full inventory state, the reasoning provided ("strategy_fit", "survival_vs_optimization"), and the final action chosen.
4.  **Debug Visualization** (`debug_viz_*.mp4`):
    - An "Explainable AI" overlay showing exactly what the bot sees.
    - Includes bounding boxes for enemies (Red) and gems (Blue), as well as the computed vector lines for navigation.

## Configuration

The bot is fully configurable via `config.yaml`.

- **Game/Window**: Set the target window name (`window_name`) and capture dimensions.
- **LLM**: Configure your provider (e.g., `gemini/gemini-3-flash`) and API keys.
- **Pilot**: Tweak force multipliers (`repel_monster`, `attract_target`) to adjust how aggressive or evasive the bot is.
- **Paths**: Update paths to your `.pt` model files if you retrain them.

## Usage

1. **Start the Game**: Open Vampire Survivors.
2. **Run the Bot**:
   ```bash
   python main.py
   ```
3. **Switch to Game**: The bot will automatically look for the "Vampire Survivors" window.

### Controls (Visualization Window)
Make sure the "Model Vision" window is in focus to use these commands.

- **`Q`**: **Recenter View**. Moves the bot's "screen capture" area to your current mouse cursor position. Useful if the window moves.
- **`P`**: **Pause/Resume**. Toggles the bot's active control.
- **`ESC`**: **Quit**. Safely stops the bot and releases controller references.

## Troubleshooting

- **Bot not moving?**: Ensure `ViGEmBus` drivers are installed. The bot creates a virtual Xbox 360 controller; check `joy.cpl` (Game Controllers) in Windows to see if it's connected.
- **Low FPS?**: Ensure you are running on a GPU. CPU inference for two YOLO models at 60FPS is difficult.
- **Screen capture is black/wrong**: Press `Q` while hovering over the game window to reset the capture region.

## Credits

This project operates as a highly modified fork of the original work by **Victor Coelho**.
- **Original Project**: [victorcoelh/vampire-survivors-bot](https://github.com/victorcoelh/vampire-survivors-bot)

We gratefully acknowledge their work, particularly the **YOLOv8 Gem/Item model**, which this project continues to use for its robust object tracking capabilities.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
