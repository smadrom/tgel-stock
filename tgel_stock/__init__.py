"""TGEL rolling-stock generator package. Runs inside Blender's bundled Python."""

BLENDER_REQUIRED_VERSION = "5.1.2"

FORBIDDEN_PATH_TOKENS = ("rc2", "legacy", "exportedproject", "models_glb")


def assert_clean_path(path: str) -> None:
    lowered = path.replace("\\", "/").lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in lowered:
            raise ValueError(f"Forbidden legacy path token '{token}' in: {path}")
