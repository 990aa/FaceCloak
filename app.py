"""HuggingFace Spaces entrypoint."""

from uacloak.interface import APP_CSS, APP_THEME, demo  # noqa: F401


if __name__ == "__main__":
    demo.launch(theme=APP_THEME, css=APP_CSS)

