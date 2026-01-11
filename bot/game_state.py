from typing import List, Dict, Any
import difflib

class GameState:
    def __init__(self):
        self.weapons: List[str] = []
        self.passives: List[str] = []
        self.max_weapons = 6
        self.max_passives = 6
        self.history: List[Dict[str, Any]] = []

        # Load knowledge base for type inference
        self.item_db = {} 
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        try:
            import json
            import os
            # Assume running from root
            path = "bot/knowledge_base/items.json"
            if os.path.exists(path):
                with open(path, "r", encoding='utf-8') as f:
                    self.item_db = json.load(f)
                print(f"[GameState] Loaded {len(self.item_db)} items from KB.")
            else:
                print(f"[GameState] Warning: Item DB not found at {path}")
                
        except Exception as e:
            print(f"[GameState] Error loading knowledge base: {e}")

    def add_weapon(self, name: str):
        if name not in self.weapons and len(self.weapons) < self.max_weapons:
            self.weapons.append(name)

    def add_passive(self, name: str):
        if name not in self.passives and len(self.passives) < self.max_passives:
            self.passives.append(name)

    def log_decision(self, decision: Dict[str, Any]):
        """
        Logs a decision made by the LLM.
        Expected structures: 
        { "action": "select", "slot": 1, "item_name": "Garlic", "reasoning": "..." }
        """
        self.history.append(decision)
        
        # Auto-update inventory if action was select
        if decision.get("action") == "select" and decision.get("item_name"):
            item_name = decision["item_name"]
            
            # Infer type if not provided
            # Infer type if not provided
            item_type = decision.get("item_type")
            canonical_name = item_name # Default to LLM's name

            if not item_type:
                if item_name in self.item_db:
                    item_type = self.item_db[item_name]
                else:
                    # Fuzzy match
                    matches = difflib.get_close_matches(item_name, self.item_db.keys(), n=1, cutoff=0.6)
                    if matches:
                        canonical_name = matches[0]
                        item_type = self.item_db[canonical_name]
                        # print(f"[GameState] Fuzzy matched '{item_name}' -> '{canonical_name}' ({item_type})")
                    else:
                        item_type = "unknown"
                        # print(f"[GameState] Inferred type for '{item_name}': {item_type}")
            
            if item_type == "weapon":
                self.add_weapon(canonical_name)
                print(f"[GameState] Added weapon: {canonical_name}")
            elif item_type == "passive":
                self.add_passive(canonical_name)
                print(f"[GameState] Added passive: {canonical_name}")
            else:
                print(f"[GameState] Unknown item type for '{item_name}'. Not added to inventory.")

    def to_json(self):
        return {
            "weapons": self.weapons,
            "passives": self.passives,
            "history_count": len(self.history)
        }
