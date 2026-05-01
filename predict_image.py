import argparse
import json
from pathlib import Path

import tensorflow as tf

from ui_helpers import PROJECT_ROOT, info, pick_file, warn


def load_class_names(class_names_path: Path):
    with open(class_names_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image_path: Path, img_size):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, axis=0)
    return arr


def main():
    base_dir = PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Predict brain tumor class for one image")
    parser.add_argument("--image", type=str, required=False, help="Path to image file")
    parser.add_argument(
        "--model",
        type=str,
        default=str(base_dir / "models" / "model_out" / "final_model.keras"),
        help="Path to trained .keras model",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        default=str(base_dir / "models" / "model_out" / "class_names.json"),
        help="Path to class_names.json",
    )
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    if not args.image:
        result = pick_file(
            "Select Image",
            base_dir / "outputs" / "images",
            [("Image files", "*.jpg;*.jpeg;*.png;*.bmp")],
        )
        if result.cancelled:
            warn("No image selected. Exiting.")
            return
        image_path = Path(result.value)
    else:
        image_path = Path(args.image)
    model_path = Path(args.model)
    class_names_path = Path(args.class_names)

    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(class_names_path)

    img_size = (args.img_size, args.img_size)
    img = preprocess_image(image_path, img_size)

    preds = model.predict(img, verbose=0)[0]
    top_idx = int(tf.argmax(preds).numpy())
    label = class_names[top_idx]
    confidence = float(preds[top_idx])

    msg = f"Label: {label}\nConfidence: {confidence:.4f}"
    print(msg)
    info(msg, title="Prediction")


if __name__ == "__main__":
    main()
