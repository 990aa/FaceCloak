"""HuggingFace Spaces entrypoint."""

from uacloak.interface import demo  # noqa: F401 — Gradio auto-discovers `demo`


if __name__ == "__main__":
    demo.launch()
