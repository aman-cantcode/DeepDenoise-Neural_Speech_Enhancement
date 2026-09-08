import os
import sys
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from model.unet       import build_unet, match_size
from audio.stft_utils import wav_to_mag_phase
from audio.enhance_utils import load_weights
from data.dataset     import make_dataset


NOISY_DIR            = str(BASE_DIR / "dataset/train/noisy")
CLEAN_DIR            = str(BASE_DIR / "dataset/train/clean")
VALID_NOISY_DIR      = str(BASE_DIR / "dataset/valid/noisy")
VALID_CLEAN_DIR      = str(BASE_DIR / "dataset/valid/clean")
BATCH_SIZE           = 4
LEARNING_RATE        = 1e-4
NUM_EPOCHS           = 30
CHECKPOINT_INTERVAL  = 5
GRADIENT_CLIP_NORM   = 1.0
WEIGHTS_DIR          = str(BASE_DIR / "weights")
SEED                 = 42



@tf.function
def train_step(model, optimizer, noisy_batch, clean_batch):

    with tf.GradientTape() as tape:
        noisy_mag, _ = wav_to_mag_phase(noisy_batch)
        clean_mag, _ = wav_to_mag_phase(clean_batch)

        # Add channel dimension: [B, T, F] → [B, T, F, 1] model expects 4D input for Conv2D layers
        noisy_input  = tf.expand_dims(noisy_mag, axis=-1)
        clean_target = tf.expand_dims(clean_mag, axis=-1)

        pred_mag = model(noisy_input, training=True)

        pred_mag, clean_target = match_size(pred_mag, clean_target)

        loss = tf.reduce_mean(tf.abs(pred_mag - clean_target))

    gradients = tape.gradient(loss, model.trainable_variables)

    gradients, _ = tf.clip_by_global_norm(gradients, GRADIENT_CLIP_NORM)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def valid_step(model, noisy_batch, clean_batch):

    noisy_mag, _ = wav_to_mag_phase(noisy_batch)
    clean_mag, _ = wav_to_mag_phase(clean_batch)

    noisy_input  = tf.expand_dims(noisy_mag, axis=-1)
    clean_target = tf.expand_dims(clean_mag, axis=-1)

    pred_mag = model(noisy_input, training=False)

    pred_mag, clean_target = match_size(pred_mag, clean_target)

    loss = tf.reduce_mean(tf.abs(pred_mag - clean_target))

    return loss


def train():
    tf.random.set_seed(SEED)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # Grow VRAM usage as needed instead of grabbing it all upfront
        tf.config.experimental.set_memory_growth(gpu, True)

    device_name = f"{len(gpus)} GPU(s)" if gpus else "CPU"
    print("=" * 60)
    print(f"  Device:        {device_name}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs:        {NUM_EPOCHS}")
    print("=" * 60)

    print("\nLoading dataset...")
    train_dataset = make_dataset(
        noisy_dir=NOISY_DIR,
        clean_dir=CLEAN_DIR,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_dataset = make_dataset(
        noisy_dir=VALID_NOISY_DIR,
        clean_dir=VALID_CLEAN_DIR,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    if tf.data.experimental.cardinality(train_dataset).numpy() == 0:
        raise ValueError("Training dataset contains no batches")
    if tf.data.experimental.cardinality(valid_dataset).numpy() == 0:
        raise ValueError("Validation dataset contains no batches")

    model     = build_unet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


    # Initialize weights with a dummy forward pass (TF lazy-initializes layers)
    dummy = tf.zeros([1, 497, 257, 1], dtype=tf.float32)
    model(dummy, training=False)
    print(f"  Model parameters: {model.count_params():,}")


    start_epoch = 0

    checkpoints = [
        f for f in os.listdir(WEIGHTS_DIR)
        if f.startswith("checkpoint_epoch_") and f.endswith(".weights.h5")
    ]

    if checkpoints:
        latest = max(
            checkpoints,
            key=lambda x: int(x.split("epoch_")[1].split(".")[0])
        )

        start_epoch = int(latest.split("epoch_")[1].split(".")[0])
        resume_path = os.path.join(WEIGHTS_DIR, latest)

        load_weights(model, resume_path)

        print(f"  Resumed from: {resume_path}")
        print(f"  Starting at epoch {start_epoch + 1}")


    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    # running average losses, reset every epoch
    loss_metric     = tf.keras.metrics.Mean()  
    val_loss_metric = tf.keras.metrics.Mean()   
    best_val_loss   = float("inf")

    for epoch in range(start_epoch, NUM_EPOCHS):
        loss_metric.reset_state()

        progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1:3d}/{NUM_EPOCHS}")

        for noisy_batch, clean_batch in progress_bar:
            batch_loss = train_step(model, optimizer, noisy_batch, clean_batch)
            loss_metric.update_state(batch_loss)
            progress_bar.set_postfix({"loss": f"{batch_loss.numpy():.4f}"})

        avg_loss = loss_metric.result().numpy()

        # Run validation: no gradient updates, BatchNorm in inference mode
        val_loss_metric.reset_state()
        for noisy_batch, clean_batch in valid_dataset:
            batch_val_loss = valid_step(model, noisy_batch, clean_batch)
            val_loss_metric.update_state(batch_val_loss)

        avg_val_loss = val_loss_metric.result().numpy()

        print(f"  Epoch {epoch + 1:3d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(WEIGHTS_DIR, "unet_tf_weights_best.weights.h5")
            model.save_weights(best_path)
            print(f"  New best val loss — saved → {best_path}")

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(WEIGHTS_DIR, f"checkpoint_epoch_{epoch + 1}.weights.h5")
            model.save_weights(ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

    final_path = os.path.join(WEIGHTS_DIR, "unet_tf_weights.weights.h5")
    model.save_weights(final_path)

    print("\n" + "=" * 60)
    print(f"  Training complete.")
    print(f"  Final weights → {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()