# Claude was used to help produce this code

from pre_processing import *
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import random
from pathlib import Path # delete
OUT_DIR = "analysis"
N_EXAMPLES = 4 # 4 real, 4 fake per plot


random.seed(42)

def try_open(paths, n):
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
            if len(imgs) == n:
                break
        except:
            continue
    return imgs
 
def load_sample_frames(frames_df, n=N_EXAMPLES):
    """Pick n random real and n random fake frames that actually open."""
    real_paths = frames_df[frames_df["label"] == "real"]["frame_filepath"].tolist()
    fake_paths = frames_df[frames_df["label"] == "fake"]["frame_filepath"].tolist()
 
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
 
    return try_open(real_paths, n), try_open(fake_paths, n)
 
 
# ── Plot 1: FFT magnitude spectrum ──────────────────────────────────────────
# Shows the frequency content of each image.
# Real images have smooth spectra; GAN-generated images often show
# a faint cross or grid pattern from the upsampling layers.
 
def plot_fft(real_imgs, fake_imgs):
    n = min(len(real_imgs), len(fake_imgs))
    fig, axes = plt.subplots(4, n, figsize=(n * 3, 10))
    fig.suptitle("FFT Magnitude Spectra — Real vs Fake\n"
                 "Look for grid/cross patterns in fake spectra (absent in real)",
                 fontsize=11)
 
    for i in range(n):
        for row, img, label in [(0, real_imgs[i], "Real"), (2, fake_imgs[i], "Fake")]:
            gray = np.array(img.convert("L")).astype(float)
            mag  = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
 
            axes[row,   i].imshow(img)
            axes[row,   i].axis("off")
            axes[row+1, i].imshow(mag, cmap="inferno")
            axes[row+1, i].axis("off")
 
            if i == 0:
                axes[row,   0].set_ylabel(f"{label}\nimage",    fontsize=9)
                axes[row+1, 0].set_ylabel(f"{label}\nspectrum", fontsize=9)
 
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fft_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_DIR}/fft_comparison.png")
 
 
# ── Plot 2: Noise residual ───────────────────────────────────────────────────
# Subtracts a blurred version of the image to isolate the noise layer.
# Real images show random, even noise. Fake images often show patterned
# or spatially uneven noise, especially around face boundaries.
 
def plot_noise_residual(real_imgs, fake_imgs):
    n = min(len(real_imgs), len(fake_imgs))
    fig, axes = plt.subplots(4, n, figsize=(n * 3, 10))
    fig.suptitle("Noise Residual — Real vs Fake\n"
                 "Uneven or patterned noise (esp. at face edges) suggests synthesis",
                 fontsize=11)
 
    for i in range(n):
        for row, img, label in [(0, real_imgs[i], "Real"), (2, fake_imgs[i], "Fake")]:
            residual = np.array(img).astype(float) - \
                       np.array(img.filter(ImageFilter.GaussianBlur(3))).astype(float)
            residual_vis = np.clip(residual * 5 + 128, 0, 255).astype(np.uint8)
 
            axes[row,   i].imshow(img)
            axes[row,   i].axis("off")
            axes[row+1, i].imshow(residual_vis)
            axes[row+1, i].axis("off")
 
            if i == 0:
                axes[row,   0].set_ylabel(f"{label}\nimage",    fontsize=9)
                axes[row+1, 0].set_ylabel(f"{label}\nresidual", fontsize=9)
 
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/noise_residual.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_DIR}/noise_residual.png")
 
 
# ── Plot 3: Pixel intensity distributions ───────────────────────────────────
# Histograms of R, G, B channel values across real vs fake frames.
# Subtle shifts in channel distributions can reflect how GANs
# model colour differently from real cameras.
 
def plot_channel_histograms(real_imgs, fake_imgs):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("RGB Channel Distributions — Real vs Fake\n"
                 "Shifts suggest GAN colour modelling differs from real cameras",
                 fontsize=11)
 
    channel_names = ["Red", "Green", "Blue"]
    colours       = ["red", "green", "blue"]
 
    for c, (name, colour) in enumerate(zip(channel_names, colours)):
        for imgs, label, alpha in [(real_imgs, "Real", 0.6), (fake_imgs, "Fake", 0.6)]:
            # Stack all example images and take the channel
            pixels = np.concatenate([np.array(img)[:, :, c].flatten() for img in imgs])
            axes[c].hist(pixels, bins=64, range=(0, 255), alpha=alpha,
                         label=label, color=colour if label == "Real" else "grey",
                         density=True)
 
        axes[c].set_title(name)
        axes[c].set_xlabel("Pixel value")
        axes[c].legend(fontsize=8)
 
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/channel_histograms.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_DIR}/channel_histograms.png")
 



if __name__ == "__main__":
    frames_df = get_frames(subset=False, save_imgs=False)

    real_imgs, fake_imgs = load_sample_frames(frames_df)
 
    print("Generating plots...")
    plot_fft(real_imgs, fake_imgs)
    plot_noise_residual(real_imgs, fake_imgs)
    plot_channel_histograms(real_imgs, fake_imgs)
 
    print(f"\nDone. Three plots saved to {OUT_DIR}/")


