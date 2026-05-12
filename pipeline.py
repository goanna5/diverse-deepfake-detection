# STEPS:
# - apply techniques (compression, blur, resize)
# - load models from hugging face
# - run combo of each frame * each model * each technique
# - thats actually quite alot of combos = 160000ish * 5 * 3 * 3... oops

from PIL import Image, ImageFilter
import io
from pre_processing import *

# Image degradations/corruptions/augmentation
COMPRESSION_QUALITY = 25
BLUR_RADIUS = 2.0
RESIZE_SCALE = 0.5

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

# deepfake detection models



if __name__ == "__main__":

    frames = extract_all_img_frames(setup(), False)
    print(frames.iloc[0, 5])
    print(type(Image.open(frames.iloc[0, 5])))
    print(type(frames.iloc[0, 7]))