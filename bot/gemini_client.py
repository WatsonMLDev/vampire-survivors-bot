import os
import json
import logging
import time
from typing import Optional, Dict, Any
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        load_dotenv() # Load variables from .env
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        
        # Load evolution knowledge base
        self.evolutions = []
        try:
            with open("bot/knowledge_base/evolutions.json", "r") as f:
                self.evolutions = json.load(f)
        except FileNotFoundError:
            logger.warning("evolutions.json not found. Running without evolution context.")

        self.sample_strategy = []
        try:
            with open("bot/knowledge_base/sample_strategy.json", "r") as f:
                self.sample_strategy = json.load(f)
        except FileNotFoundError:
            logger.warning("sample_strategy.json not found. Running without sample strategy context.")

    def get_decision(self, frame_image: Image.Image, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sends the screenshot and game state to Gemini to get a level-up decision.
        """
        if not self.api_key:
            return None

        prompt = self._construct_prompt(game_state)
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, frame_image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
            
            # 1. If result is a list (Gemini sometimes returns [decision]), take the first item
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], dict):
                    result = result[0]
                else:
                    logger.error(f"Gemini returned an unexpected list format: {result}")
                    return None

            # 2. Handle nested 'decision' structure (Grand Strategist format)
            if isinstance(result, dict) and "decision" in result:
                logger.info(f"Gemini Analysis: {result.get('analysis')}")
                return result["decision"]
            
            # 3. Fallback: Assume the dict itself is the decision (Legacy format)
            if isinstance(result, dict):
                return result

            return None
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return None

    def _construct_prompt(self, game_state: Dict[str, Any]) -> str:
        inventory_str = json.dumps(game_state, indent=2)
        history_str = json.dumps(game_state.get("history", []), indent=2)
        evolutions_str = json.dumps(self.evolutions[:50], indent=2) # Limit context if large
        sample_strategy_str = json.dumps(self.sample_strategy, indent=2)
        
        # Calculate counts
        current_weapon_count = len(game_state.get("weapons", []))
        current_passive_count = len(game_state.get("passives", []))

        return f"""
        You are the Grand Strategist AI for Vampire Survivors. Your objective is to build an invincible endgame loadout by mastering "Slot Economy" and "Pool Pruning."

        ### 1. CURRENT GAME STATE
        - **Inventory**: {inventory_str}
        - **Empty Slots**: {6 - current_weapon_count} Weapons | {6 - current_passive_count} Passives
        - **Available Resources**: Use the image to find the amount of Reroll, Skip, Banish left
        - **Decision History**: {history_str}

        ### 2. STRATEGIC INTELLIGENCE
        - **Samples of Target Builds/Strategies you COULD follow**: {sample_strategy_str}
        - **Evolution & Union Chart**: {evolutions_str}
        - **The Union Rule**: Remember that Unions (e.g., Peachone + Ebony Wings) merge two weapons into ONE slot, liberating a slot for a future weapon.

        ### 3. DECISION HIERARCHY (Priority Order)
        1. **Evolution/Union Completion**: If an option completes a weapon/passive pair you already own, take it immediately.
        2. **Core Component Acquisition**: If an option is a "Best-in-Slot" weapon or passive for your target strategy AND you have an open slot.
        3. **Incremental Upgrade**: If no new core items appear, upgrade a weapon you already own (e.g., Level 3 -> 4). This is usually better than skipping if Threat Level is Medium or High.
        4. **Strategic Pruning (Banish/Skip)**: 
            - Use **Banish** on "trash" items that don't fit the strategy to remove them from the game permanently.
            - Use **Skip/Reroll** if you must keep a slot open for a specific catalyst you haven't found yet.


        ### 4. THE TASK
        Analyze the number of options on the Level Up screen. You must decide: 
        Is it better to **COMMIT** to a new item, **UPGRADE** an existing one, or **PRUNE** the pool to protect your future build?

        **Output Format (JSON)**:
        {{
            "analysis": {{
                "strategy_fit": "Does anything on screen complete an evolution or fit the long-term plan?",
                "slot_management": "How many slots remain, and can a Union free one up later?",
                "survival_vs_optimization": "Comparison of the value of an incremental upgrade vs. the value of a Banish/Skip."
            }},
            "decision": {{
                "action": "select" | "reroll" | "skip" | "banish",
                "slot": int (1-4),
                "item_name": "Name of item",
                "reasoning": "A concise explanation of the strategic choice."
            }}
        }}
        """
