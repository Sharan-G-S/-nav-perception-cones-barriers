"""Target classes, their display names and real-world heights in metres."""

CLASS_NAMES = ["cone", "barrier", "stop_sign"]

DISPLAY_NAMES = {"cone": "Cone", "barrier": "Barrier", "stop_sign": "Stop Sign"}

OBJECT_HEIGHTS_M = {"cone": 0.70, "barrier": 1.00, "stop_sign": 0.75}


def class_index(name: str) -> int:
    return CLASS_NAMES.index(name)


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())
