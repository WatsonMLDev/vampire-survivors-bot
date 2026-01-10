from typing import List, Dict, Any

class GameState:
    def __init__(self):
        self.weapons: List[str] = []
        self.passives: List[str] = []
        self.max_weapons = 6
        self.max_passives = 6
        self.history: List[Dict[str, Any]] = []

    def add_weapon(self, name: str):
        if name not in self.weapons and len(self.weapons) < self.max_weapons:
            self.weapons.append(name)
        # In a real scenario, we might track levels too, but for now just names

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
            item_type = decision.get("item_type", "unknown") # 'weapon' or 'passive'
            
            # Simple heuristic if type isn't provided, though ideally Gemini tells us
            # For now, we rely on Gemini telling us, or valid updates from `main.py`
            if item_type == "weapon":
                self.add_weapon(item_name)
            elif item_type == "passive":
                self.add_passive(item_name)
    
    def update_from_treasure(self, screenshot):
        """
        Stub: Analyze screenshot to identify items gained from treasure chest.
        """
        # In the future, send 'screenshot' to Gemini to identify the item.
        print("Treasure processed (Stub).")

    def to_json(self):
        return {
            "weapons": self.weapons,
            "passives": self.passives,
            "history_count": len(self.history)
        }
