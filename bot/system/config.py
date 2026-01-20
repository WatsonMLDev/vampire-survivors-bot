import yaml
import os

class Config:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Look for config.yaml in root or current dir
        paths_to_check = [
            'config.yaml',
            os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        ]
        
        for path in paths_to_check:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self._config = yaml.safe_load(f)
                return
        
        if self._config is None:
             # Fallback warning
             pass
             self._config = {}

    def get(self, path=None, default=None):
        if not path:
            return self._config
            
        keys = path.split('.')
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

# Global instance
config = Config()
