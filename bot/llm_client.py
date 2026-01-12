import os
import json
import logging
from typing import Optional, Dict, Any, List
from PIL import Image
from dotenv import load_dotenv
import base64
import io
from litellm import completion
import time

# Load environment variables
load_dotenv()

# Suppress LiteLLM logging
os.environ["LITELLM_LOG"] = "WARNING"

# Configure logging
logging.basicConfig(level=logging.INFO)
# Quiet down third party libs
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING) # Case sensitive fix
logger = logging.getLogger(__name__)

from bot.config import config
import warnings

# Suppress Pydantic serialization warnings typically caused by LiteLLM interactions
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

class LLMClient:
    def __init__(self):
        self.model_name = config.get("llm.model", "gemini/gemini-1.5-flash")
        self.api_key_env_var = config.get("llm.api_key_env_var", "GOOGLE_API_KEY")
        self.api_base = config.get("llm.api_base", None)
        
        self.api_key = os.environ.get(self.api_key_env_var)
        if not self.api_key and "ollama" not in self.model_name: # Ollama might not need a key
             logger.warning(f"{self.api_key_env_var} not found in environment variables.")

        # Load evolution knowledge base
        self.evolutions = []
        try:
            with open("bot/knowledge_base/evolutions.json", "r") as f:
                self.evolutions = json.load(f)
        except FileNotFoundError:
            logger.warning("evolutions.json not found. Running without evolution context.")

        try:
            with open("bot/knowledge_base/sample_strategy.json", "r") as f:
                self.sample_strategy = json.load(f)
        except FileNotFoundError:
            logger.warning("sample_strategy.json not found. Running without sample strategy context.")

        # Logging Setup
        self.log_enabled = True # Could drive from config
        self.output_dir = config.get('capture.output_dir', 'training_data')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.output_dir, f"decisions_{timestamp}.jsonl")

        # Initialize Static System Content for Caching
        self.static_system_content = self._get_system_content()

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _get_system_content(self) -> str:
        """Returns the static part of the prompt that can be cached."""
        evolutions_str = json.dumps(self.evolutions[:50], indent=2) # Limit context if large
        sample_strategy_str = json.dumps(self.sample_strategy, indent=2)

        return f"""
        You are the Grand Strategist AI for Vampire Survivors. Your objective is to build an invincible endgame loadout by mastering "Slot Economy" and "Pool Pruning."

        ### STRATEGIC INTELLIGENCE
        - **Samples of Target Builds/Strategies you COULD follow**: {sample_strategy_str}
        - **Evolution & Union Chart**: {evolutions_str}
        - **The Union Rule**: Remember that Unions (e.g., Peachone + Ebony Wings) merge two weapons into ONE slot, liberating a slot for a future weapon.

        ### DECISION HIERARCHY (Priority Order)
        1. **Evolution/Union Completion**: If an option completes a weapon/passive pair you already own, take it immediately.
        2. **Core Component Acquisition**: If an option is a "Best-in-Slot" for your strategy, take it.
        3. **Slot Filling (Early Game)**: If you have > 3 empty slots, favor taking ANY decent weapon/passive over Banishing, unless it is actively harmful (e.g. Curse) or worthless. Do not be too picky early on; survival requires firepower.
        4. **Incremental Upgrade**: Upgrade existing items if no new useful items appear.
        5. **Strategic Pruning**: Only use Banish if:
            - The item is strictly harmful (like Skull O'Maniac if weak).
            - You have < 2 slots left and MUST save them for a specific key component.
            - The item appears repeatedly and you never want it.

        ### THE TASK
        Analyze the number of options on the Level Up screen. You must decide: 
        Is it better to **COMMIT** to a new item, **UPGRADE** an existing one, or **PRUNE** the pool to protect your future build?

        **Output Format (JSON)**:
        {{
            "analysis": {{
                "visible_options": ["List of items seen on screen (Top to Bottom)"],
                "strategy_fit": "Does anything on screen complete an evolution or fit the long-term plan?",
                "slot_management": "How many slots remain, and can a Union free one up later?",
                "survival_vs_optimization": "Comparison of the value of an incremental upgrade vs. the value of a Banish/Skip."
            }},
            "decision": {{
                "action": "select" | "reroll" | "skip" | "banish",
                "slot": int (1-4),
                "item_name": "Name of item"
            }}
        }}
        """

    def _get_user_content(self, game_state: Dict[str, Any]) -> str:
        """Returns the dynamic part of the prompt based on current game state."""
        inventory_str = json.dumps(game_state, indent=2)
        
        # Limit history to prevent context pollution and prune unnecessary fields
        full_history = game_state.get("history", [])
        recent_history = full_history[-15:] if len(full_history) > 15 else full_history
        
        # Simplified history: Drop 'slot' and 'reasoning'
        simplified_history = [
            {"action": h.get("action"), "item": h.get("item_name")} 
            for h in recent_history
        ]
        history_str = json.dumps(simplified_history, indent=2)

        # Calculate counts
        current_weapon_count = len(game_state.get("weapons", []))
        current_passive_count = len(game_state.get("passives", []))

        return f"""
        ### CURRENT GAME STATE (What you ALREADY OWN)
        - **Inventory**: {inventory_str}
        - **Empty Slots**: {6 - current_weapon_count} Weapons | {6 - current_passive_count} Passives
        - **Available Resources**: Use the image to find the amount of Reroll, Skip, Banish left
        - **Decision History**: {history_str}

        ### VISUAL ANALYSIS (What is on SCREEN)
        - Look at the provided screenshot. These are the *ONLY* options you can choose from.
        - **CRITICAL**: Do not confuse items in your Inventory with items on the Screen. 
        - If you see an item on screen that you already have, selecting it means **UPGRADING** it.
        - If you see an item you don't have, selecting it means **ACQUIRING** it.
        - **Evolved Weapons** (like Holy Wand, Death Spiral) NEVER appear on Level Up screens. If you think you see one, look closer; it's likely the base weapon or something else.
        """

    def get_decision(self, frame_image: Image.Image, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sends the screenshot and game state to the LLM to get a level-up decision.
        Uses Context Caching for the system prompt.
        """
        user_prompt = self._get_user_content(game_state)
        
        # Prepare content for LiteLLM (Standard OpenAI Multimodal Format)
        base64_image = self._image_to_base64(frame_image)
        image_url = f"data:image/jpeg;base64,{base64_image}"
        
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text", 
                        "text": self.static_system_content,
                        "cache_control": {"type": "ephemeral"} 
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]
        
        # Schema Definition (OpenAI Structured Outputs Format)
        decision_schema = {
            "name": "game_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "object",
                        "properties": {
                            "visible_options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List the names of the 3 or 4 items visible in the screenshot, from top to bottom."
                            },
                            "strategy_fit": {"type": "string"},
                            "slot_management": {"type": "string"},
                            "survival_vs_optimization": {"type": "string"}
                        },
                        "required": ["visible_options", "strategy_fit", "slot_management", "survival_vs_optimization"],
                        "additionalProperties": False
                    },
                    "decision": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["select", "reroll", "skip", "banish"]},
                            "slot": {"type": "integer"},
                            "item_name": {"type": "string"}
                        },
                        "required": ["action", "slot", "item_name"],
                        "additionalProperties": False
                    }
                },
                "required": ["analysis", "decision"],
                "additionalProperties": False
            }
        }

        # LiteLLM Arguments
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "response_format": { "type": "json_schema", "json_schema": decision_schema } 
        }
        
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        if self.api_base:
            kwargs["api_base"] = self.api_base

        try:
            response = completion(**kwargs)
            
            # Log usage for verification if available
            token_usage_Log = {}
            if hasattr(response, 'usage'):
                usage = response.usage
                # logger.info(f"Token Usage: {usage}")
                
                # Extract Usage Stats safely
                token_usage_Log = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                    "cached_tokens": 0
                }
                
                # Handle Prompt Token Details for Caching
                if hasattr(usage, "prompt_tokens_details"):
                     details = usage.prompt_tokens_details
                     token_usage_Log["cached_tokens"] = getattr(details, "cached_tokens", 0)

            content = response.choices[0].message.content   
            result = json.loads(content)
            
            # Schema guarantees 'decision' key exists and is valid
            if "decision" in result:
                logger.info(f"LLM Analysis: {result['decision']}")
                
                # [NEW] Log to JSONL
                if self.log_enabled:
                    try:
                        inventory_str = json.dumps(game_state, indent=2)
                        log_entry = {
                            "timestamp": time.time(),
                            "inventory_str": inventory_str,
                            "llm_output": result,
                            "token_usage": token_usage_Log 
                        }
                        with open(self.log_filename, "a", encoding='utf-8') as f:
                            json.dump(log_entry, f)
                            f.write('\n')
                    except Exception as loc_e:
                        logger.error(f"Failed to log LLM decision: {loc_e}")

                return result["decision"]
            
            logger.error(f"Schema violation: 'decision' key missing in {result}")
            return None
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
