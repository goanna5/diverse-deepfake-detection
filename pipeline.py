# STEPS:
# - apply techniques (compression, blur, resize)
# - load models from hugging face
# - run combo of each frame * each model * each technique
# - thats actually quite alot of combos = 160000ish * 5 * 3 * 3... oops

from PIL import Image, ImageFilter
import io
from pre_processing import *
import timm
import torch
from torchvision import transforms

# Image degradations/corruptions/augmentation
COMPRESSION_QUALITY = 25
BLUR_RADIUS = 2.0
RESIZE_SCALE = 0.5

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

AUGMENTATION_TECHNIQUES = {
    "clean": clean,
    "compression": compress,
    "blur": blur,
    "resize": resize,
}

# deepfake detection models used Claude to assist with the code for this

## double check xception after gaining access
def load_xception(checkpoint_path: str):
    """loads XceptionNet with FF++ checkpoint path for weights
    """
    model = timm.create_model("xception", pretrained=True, num_classes=2)
    state = torch.load(checkpoint_path, map_location=DEVICE)

    # FaceForensics++ checkpoints are sometimes wrapped in a dict
    if "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=False)
    model.eval()
    print("XceptionNet loaded!")
    return model.to(DEVICE)

def load_efficientnet():
    """EfficientNet-B4 from timm with pretrained weights from ImageNet
    """
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=2)
    model.eval()
    print("EfficientNet-B4 loaded!")
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




if __name__ == "__main__":

    # frames = get_frames(subset=True, save_imgs=False)
    # with Image.open(frames.iloc[0]['frame_filepath']) as im:
    #     new = resize(im)
    eff_net = load_efficientnet()
    print(get_transform_for_timm(eff_net))
    # print(frames.iloc[0, 5])
    # print(type(Image.open(frames.iloc[0, 5])))
    # print(type(frames.iloc[0, 7]))