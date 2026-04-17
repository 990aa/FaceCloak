"""HuggingFace Spaces entrypoint."""

import os

from uacloak.interface import APP_CSS, APP_THEME, demo  # noqa: F401


if __name__ == "__main__":
    is_space = bool(os.getenv("SPACE_ID"))

    launch_kwargs: dict[str, object] = {
        "theme": APP_THEME,
        "css": APP_CSS,
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
