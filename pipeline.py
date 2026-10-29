"""Image degradations, datasets, models and fine-tuning for deepfake detection.
"""

import io

import numpy as np
import timm
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from pre_processing import *

# Image degradation / corruption / augmentation parameters
COMPRESSION_QUALITY = 25
BLUR_RADIUS = 2.0
RESIZE_SCALE = 0.5
NOISE_SIGMA = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean(img: Image.Image) -> Image.Image:
    """Return an unchanged copy of the image.

    Args:
        img: Pillow image.

    Returns:
        A copy of ``img`` with no modifications.
    """
    return img.copy()


def compress(img: Image.Image,
             quality: int = COMPRESSION_QUALITY) -> Image.Image:
    """Apply JPEG compression; pixel values change, dimensions stay the same.

    Args:
        img: Pillow image.
        quality: JPEG quality factor, from 1 (worst) to 95 (best).

    Returns:
        The compressed image.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def blur(img: Image.Image, radius: float = BLUR_RADIUS) -> Image.Image:
    """Apply a Gaussian blur to the image.

    Args:
        img: Pillow image.
        radius: Standard deviation of the Gaussian kernel.

    Returns:
        The blurred image.
    """
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def resize(img: Image.Image, scale: float = RESIZE_SCALE) -> Image.Image:
    """Downscale the image, then upscale it back to its original size.

    Args:
        img: Pillow image.
        scale: Factor by which the image is downscaled (e.g. 0.5 halves
            each dimension).

    Returns:
        The degraded image, at the same size as the input.
    """
    width, height = img.size
    small_size = (int(width * scale), int(height * scale))
    downscaled = img.resize(small_size, Image.LANCZOS)
    return downscaled.resize((width, height), Image.LANCZOS)


def gaussian_noise(img: Image.Image,
                   sigma: float = NOISE_SIGMA) -> Image.Image:
    """Add Gaussian noise to the image.

    Args:
        img: Pillow image (converted to RGB).
        sigma: Standard deviation of the noise, in pixel intensity units.

    Returns:
        The noisy RGB image, with pixel values clipped to ``[0, 255]``.
    """
    img_array = np.array(img.convert("RGB")).astype(np.float32)
    noise = np.random.normal(0, sigma, img_array.shape)
    noise_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noise_array)


AUGMENTATION_TECHNIQUES = {
    "clean": clean,
    "compression": compress,
    "blur": blur,
    "resize": resize,
    "noise": gaussian_noise,
}

LABEL_MAP = {
    "real": 0,
    "fake": 1,
}


class Dataset(torch.utils.data.Dataset):
    """Dataset of video frames with a degradation technique applied.

    Each item is loaded from disk, degraded by ``technique``, preprocessed
    by ``transform`` and returned with its integer label.
    """

    def __init__(self, frames_df, transform, technique, label_map):
        """Initialise the dataset.

        Args:
            frames_df: DataFrame with ``frame_filepath`` and ``label``
                columns, one row per frame.
            transform: Preprocessing transform applied to each image
                after the technique.
            technique: Function taking and returning a Pillow image.
            label_map: Optional mapping applied on top of ``LABEL_MAP`` to
                remap class indices, or None to leave them unchanged.
        """
        self.frames_df = frames_df.reset_index(drop=True)
        self.transform = transform
        self.technique = technique
        self.label_map = label_map

    def __len__(self):
        """Return the number of frames in the dataset."""
        return len(self.frames_df)

    def __getitem__(self, idx):
        """Return the transformed image and label tensor at ``idx``."""
        row = self.frames_df.iloc[idx]

        img = Image.open(row["frame_filepath"]).convert("RGB")
        img = self.technique(img)
        img = self.transform(img)
        label = LABEL_MAP[row["label"]]

        if self.label_map is not None:
            label = self.label_map[label]
        label = torch.tensor(int(label), dtype=torch.long)

        return img, label


def split_by_vid(frames_df, test_portion=0.2, seed=42):
    """Split frames into train and test sets by source video.

    Splitting is done on videos before frames are used, to avoid bias from
    frames of the same video appearing in both sets. The split is made
    separately within each label so the real/fake ratio is kept equal.

    Args:
        frames_df: DataFrame with ``label`` and ``source`` columns.
        test_portion: Fraction of videos (per label) placed in the test set.
        seed: Seed for the random number generator, for reproducibility.

    Returns:
        A tuple ``(train_df, test_df)`` of DataFrames with reset indices.
    """
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for _, group in frames_df.groupby("label"):
        vids = group["source"].unique()
        rng.shuffle(vids)

        cutoff = int(len(vids) * (1 - test_portion))
        train_vids, test_vids = vids[:cutoff], vids[cutoff:]
        train_parts.append(group[group["source"].isin(train_vids)])
        test_parts.append(group[group["source"].isin(test_vids)])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


def load_xception():
    """Load XceptionNet with a two-class head.

    Returns:
        The model in eval mode, on ``DEVICE``.
    """
    model = timm.create_model("xception", pretrained=True, num_classes=2)
    model.eval()
    print("XceptionNet loaded!")
    return model.to(DEVICE)


def load_efficientnet4():
    """Load EfficientNet-B4 from timm with ImageNet pretrained weights.

    Returns:
        The model in eval mode, on ``DEVICE``.
    """
    model = timm.create_model(
        "efficientnet_b4", pretrained=True, num_classes=2
    )

    model.eval()
    print("EfficientNet-B4 loaded!")
    return model.to(DEVICE)


def load_efficientnet7():
    """Load EfficientNet-B7 from timm with ImageNet pretrained weights.

    Returns:
        The model in eval mode, on ``DEVICE``.
    """
    model = timm.create_model(
        "tf_efficientnet_b7", pretrained=True, num_classes=2
    )

    model.eval()
    print("EfficientNet-B7 loaded!")
    return model.to(DEVICE)


def load_swin():
    """Load a Swin Transformer from timm (loaded like the EfficientNets).

    Returns:
        The model in eval mode, on ``DEVICE``.
    """
    model = timm.create_model(
        "swin_base_patch4_window7_224", pretrained=True, num_classes=2
    )
    model.eval()
    print("Swin Transformer loaded!")
    return model.to(DEVICE)


def get_transform_for_timm(model):
    """Build the preprocessing transform for a timm model.

    Uses the model's default config, falling back to ImageNet statistics
    and a 224x224 input size if values are missing. As this is applied to
    all models in the same way, it should be consistent across them.

    Args:
        model: A timm model with a ``default_cfg`` attribute.

    Returns:
        A ``transforms.Compose`` that resizes, converts to a tensor and
        normalises.
    """
    cfg = model.default_cfg
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    size = cfg.get("input_size", (3, 224, 224))[-1]

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


MODELS = {
    "xception": load_xception,
    "efficientnet4": load_efficientnet4,
    "efficientnet7": load_efficientnet7,
    "swin": load_swin,
}


def freeze(model, unfreeze_last_n_params=20):
    """Freeze the model, leaving only the head and last parameters trainable.

    All parameters are frozen, then the classifier head and the last
    ``unfreeze_last_n_params`` parameters of the backbone are unfrozen.

    Args:
        model: The model to modify in place.
        unfreeze_last_n_params: Number of final parameters to unfreeze.
            If 0 or less, only the classifier head is unfrozen.

    Returns:
        The same model, with ``requires_grad`` updated.
    """
    for p in model.parameters():
        p.requires_grad = False

    for p in model.get_classifier().parameters():
        p.requires_grad = True

    if unfreeze_last_n_params > 0:
        all_params = list(model.parameters())
        for p in all_params[-unfreeze_last_n_params:]:
            p.requires_grad = True
    return model


def fine_tune(model, train_loader, test_loader, epochs=10, lr=1e-4,
              model_name="model"):
    """Fine-tune a model, saving the checkpoint with the best test accuracy.

    Args:
        model: The model to train.
        train_loader: DataLoader yielding ``(images, labels)`` for training.
        test_loader: DataLoader used to evaluate after each epoch.
        epochs: Number of training epochs.
        lr: Learning rate for the Adam optimiser.
        model_name: Name used in the checkpoint file ``best_<name>.pth``.

    Returns:
        The model after the final epoch. Note that this is not necessarily
        the best checkpoint, which is saved to disk separately.
    """
    model.to(DEVICE)
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_test_acc = 0
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimiser.step()

        test_acc = evaluate(model, test_loader)
        print(f"Epoch: {epoch + 1}/{epochs} | test_acc: {test_acc}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")

    return model


@torch.no_grad()
def evaluate(model, loader):
    """Compute the classification accuracy of a model over a DataLoader.

    Args:
        model: The model to evaluate.
        loader: DataLoader yielding ``(images, labels)``.

    Returns:
        The fraction of samples classified correctly.
    """
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def predict(model, loader):
    """Create lists of predictions and labels, in dataset order.

    Args:
        model: The model used for prediction.
        loader: DataLoader yielding ``(images, labels)``.

    Returns:
        A tuple ``(all_preds, all_labels)`` of lists of class indices.
    """
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return all_preds, all_labels