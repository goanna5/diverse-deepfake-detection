# STEPS:
# - apply techniques (compression, blur, resize)
# - load models from hugging face
# - fine tune models (hopefully)
# - run combo of each frame * each model * each technique -> actually do this in main
# - thats actually quite alot of combos = 160000ish * 5 * 3 * 3... oops

from PIL import Image, ImageFilter
import io
from pre_processing import *
import timm
import torch
from torchvision import transforms
import numpy as np

# Image degradations/corruptions/augmentation
COMPRESSION_QUALITY = 25
BLUR_RADIUS = 2.0
RESIZE_SCALE = 0.5
NOISE_SIGMA = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def clean(img: Image.Image) -> Image.Image:
    """No changes made, takes in Pillow Image and returns a copied version unchanged"""
    return img.copy()

def compress(img: Image.Image, quality: int = COMPRESSION_QUALITY) -> Image.Image:
    """jpeg compression, pixel values change, dimensions stay the same
    img: pillow Image
    quality: Q-factor, 1 (worst) to 95 (best)
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()

def blur(img: Image.Image, radius: float = BLUR_RADIUS) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def resize(img: Image.Image, scale: float = RESIZE_SCALE) -> Image.Image:
    """Could be something to consider using diff types of scaling, I use
    LANCZOS but could use the others?
    https://www.geeksforgeeks.org/python/python-pil-image-resize-method/
    """
    width, height = img.size
    small_size = (int(width * scale), int(height * scale))
    downscaled = img.resize(small_size, Image.LANCZOS)
    return downscaled.resize((width, height), Image.LANCZOS)

def gaussian_noise(img: Image.Image, sigma: float = NOISE_SIGMA) -> Image.Image:
    """ random may need to be moved to where this func is called because it may 
    not be random each time
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

# dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, frames_df, transform, technique, label_map):
        self.frames_df = frames_df.reset_index(drop=True)
        self.transform = transform
        self.technique = technique
        self.label_map = label_map 

    def __len__(self):
        return len(self.frames_df)

    def __getitem__(self, idx):
        row = self.frames_df.iloc[idx]

        img = Image.open(row["frame_filepath"]).convert("RGB")
        img = self.technique(img)
        img = self.transform(img)
        label = LABEL_MAP[row["label"]]

        if self.label_map is not None:
            label = self.label_map[label]
        label = torch.tensor(int(label), dtype=torch.long)

        return img, label

# split videos into train-test before extracting frames to avoid bias
# additionally, ensure theres equal ratio of real and fake
def split_by_vid(frames_df, test_portion=0.2, seed=42):
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


# deepfake detection models used Claude to assist with the code for this

## double check xception after gaining access
def load_xception():
    """loads XceptionNet with FF++ checkpoint path for weights
    """
    model = timm.create_model("xception", pretrained=True, num_classes=2)
    model.eval()
    print("XceptionNet loaded!")
    return model.to(DEVICE)

def load_efficientnet4():
    """EfficientNet-B4 from timm with pretrained weights from ImageNet
    """
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=2)

    model.eval()
    print("EfficientNet-B4 loaded!")
    return model.to(DEVICE)

def load_efficientnet7():
    """EfficientNet-B7 from timm with pretrained weights from ImageNet
    """
    model = timm.create_model("tf_efficientnet_b7", pretrained=True, num_classes=2)

    model.eval()
    print("EfficientNet-B7 loaded!")
    return model.to(DEVICE)

def load_swin():
    """Swin Transformer, basically same loading method as efficient net
    """
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=2)
    model.eval()
    print("Swin Transformer loaded!")
    return model.to(DEVICE)

def get_transform_for_timm(model):
    """preprocessing needed for timm (as its done to all it should be fine bc its 
    consistent??)
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

# fine tuning so the models are deepfake detectors 
# (Claude was used to help develop this)
def freeze(model, unfreeze_last_n_params=20):
    """Freeze all params, unfreeze classifier head,
    and last n params of backbone
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

def fine_tune(model, train_loader, test_loader, epochs=10, lr=1e-4, model_name="model"):
    model.to(DEVICE)
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
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
    """creates lists of predications and labels in dataset order"""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return all_preds, all_labels

if __name__ == "__main__":
    pass
    # frames = get_frames(subset=True, save_imgs=False)
    # with Image.open(frames.iloc[0]['frame_filepath']) as im:
    #     new = resize(im)
    # eff_net = load_efficientnet()
    # eff_net = freeze(eff_net, unfreeze_last_n_params=20)
    # transform = get_transform_for_timm(eff_net)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    # print(fine_tune(eff_net, train_loader, test_loader))
    # # print(frames.iloc[0, 5])
    # # print(type(Image.open(frames.iloc[0, 5])))
    # # print(type(frames.iloc[0, 7]))
