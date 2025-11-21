import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from src.model.builder import load_dataset, build_model


def main():
    X, y = load_dataset()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.5,
        random_state=11,
        stratify=y
    )

    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=y_train.shape[1]
    )

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_dir = "Models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model_2011.keras")

    callbacks = [
        TensorBoard(log_dir=log_dir),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )

    print(f"TRAINING DONE. Best saved at: {checkpoint_path}")


if __name__ == "__main__":
    main()
