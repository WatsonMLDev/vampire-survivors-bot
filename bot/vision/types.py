from collections import namedtuple

# Shared Detection type to avoid circular imports
Detection = namedtuple("Detection", ["position", "label", "confidence"])
