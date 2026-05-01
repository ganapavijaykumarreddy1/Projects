import argparse
import json
from pathlib import Path

import tensorflow as tf


def build_datasets(train_dir: Path, test_dir: Path, img_size, batch_size, seed):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes, img_size, base_trainable=False, dropout=0.2):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    x = preprocess(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def main():
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Brain tumor image classifier (local)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(base_dir / "data" / "brain tumour dataset"),
        help="Path to dataset root with Training/Testing subfolders",
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default=str(base_dir / "models" / "model_out")
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = (args.img_size, args.img_size)

    train_ds, val_ds, test_ds, class_names = build_datasets(
        train_dir, test_dir, img_size, args.batch_size, args.seed
    )

    model, base_model = build_model(
        num_classes=len(class_names),
        img_size=img_size,
        base_trainable=False,
        dropout=0.2,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if args.fine_tune:
        base_model.trainable = True
        # Keep the first 100 layers frozen for stability
        for layer in base_model.layers[:100]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save(output_dir / "final_model.keras")
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)


if __name__ == "__main__":
    main()
