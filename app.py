"""Hugging Face Spaces entrypoint."""

import os

from visioncloak.interface import demo  # noqa: F401


if __name__ == "__main__":
    is_space = bool(os.getenv("SPACE_ID"))

    launch_kwargs: dict[str, object] = {
        # SSR can be brittle behind some hosted reverse proxies.
        "ssr_mode": False,
        "show_error": True,
    }

    if is_space:
        launch_kwargs.update(
            {
                "server_name": "0.0.0.0",
                "server_port": int(os.getenv("PORT", "7860")),
                # Avoid localhost accessibility checks that can fail in containerized Spaces.
                "share": True,
            }
        )

    demo.launch(**launch_kwargs)
