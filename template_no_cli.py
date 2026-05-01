"""
Template: No-CLI script using UI helpers.
Copy this file when creating a new script.
"""

from pathlib import Path

from ui_helpers import PROJECT_ROOT, ask_str, info, pick_file, warn


def main():
    base_dir = PROJECT_ROOT
    result = pick_file(
        "Select Image",
        base_dir / "outputs" / "images",
        [("Image files", "*.jpg;*.jpeg;*.png;*.bmp")],
    )
    if result.cancelled:
        warn("No image selected. Exiting.")
        return

    name = Path(result.value).name
    tag_result = ask_str("Enter a tag:", "Tag", "example")
    if tag_result.cancelled:
        warn("No tag provided. Exiting.")
        return

    info(f"Selected: {name}\nTag: {tag_result.value}", title="Template")


if __name__ == "__main__":
    main()
