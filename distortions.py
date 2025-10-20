import cv2
import numpy as np
import random
from PIL import ImageEnhance, Image



# Utility: Convert between PIL and OpenCV
def cv2_to_pil(img): return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def pil_to_cv2(img): return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class RandomDistortion:
    def __init__(self, distortions):
        self.distortions = distortions

    def __call__(self, img):
        distortion = random.choice(self.distortions)
        distorted_img, label = distortion(img)
        return distorted_img, label


class Clean:
    def __call__(self, img):
        return img, "Clean"


class LensBlur:
    def __init__(self, ksize=11):
        self.ksize = ksize

    def __call__(self, img):
        return cv2.GaussianBlur(img, (self.ksize, self.ksize), 0), "LensBlur"


class MotionBlur:
    def __init__(self, degree=15, angle=45):
        self.degree = degree
        self.angle = angle

    def __call__(self, img):
        k = np.zeros((self.degree, self.degree))
        k[int((self.degree - 1)/2), :] = np.ones(self.degree)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((self.degree / 2, self.degree / 2), self.angle, 1.0), (self.degree, self.degree))
        k /= self.degree
        return cv2.filter2D(img, -1, k), "MotionBlur"


class Blackout:
    def __call__(self, img):
        return np.zeros_like(img), "Blackout"


class Overexposure:
    def __init__(self, factor=2.5):
        self.factor = factor

    def __call__(self, img):
        pil_img = cv2_to_pil(img)
        enhancer = ImageEnhance.Brightness(pil_img)
        return pil_to_cv2(enhancer.enhance(self.factor)), "Overexposure"


class Underexposure:
    def __init__(self, factor=0.3):
        self.factor = factor

    def __call__(self, img):
        pil_img = cv2_to_pil(img)
        enhancer = ImageEnhance.Brightness(pil_img)
        return pil_to_cv2(enhancer.enhance(self.factor)), "Underexposure"


class GaussianNoise:
    def __init__(self, mean=0, std=25):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.std, img.shape).astype(np.uint8)
        return cv2.add(img, noise), "GaussianNoise"


class Compression:
    def __init__(self, quality=5):
        self.quality = quality

    def __call__(self, img):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, enc = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(enc, 1), "Compression"


class ColorDistortion:
    def __call__(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[..., 0] = (hsv[..., 0] + 50) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), "ColorDistortion"


class Glare:
    def __call__(self, img):
        h, w = img.shape[:2]
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask, (w//2, h//3), int(w*0.2), (255, 255, 255), -1)
        return cv2.addWeighted(img, 0.8, mask, 0.5, 0), "Glare"


class Ghosting:
    def __init__(self, shift=10, alpha=0.6):
        self.shift = shift
        self.alpha = alpha

    def __call__(self, img):
        overlay = np.roll(img, self.shift, axis=1)
        return cv2.addWeighted(img, 1 - self.alpha, overlay, self.alpha, 0), "Ghosting"


class Flicker:
    def __init__(self, factor=1.8):
        self.factor = factor

    def __call__(self, img):
        pil_img = cv2_to_pil(img)
        enhancer = ImageEnhance.Brightness(pil_img)
        return pil_to_cv2(enhancer.enhance(self.factor)), "Flicker"


class FrameFreeze:
    def __call__(self, img):
        return img.copy(), "FrameFreeze"


class Obstruction:
    def __call__(self, img):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (w//3, h//3), (2*w//3, 2*h//3), 255, -1)
        obstruction = cv2.GaussianBlur(np.full_like(img, (80, 80, 80)), (51, 51), 0)
        return np.where(mask[:, :, None] == 255, obstruction, img), "Obstruction"


class Crop:
    def __call__(self, img):
        h, w = img.shape[:2]
        return cv2.copyMakeBorder(img[:h//2, :w//2], h//2, 0, w//2, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]), "Crop"


class Aliasing:
    def __init__(self, factor=4):
        self.factor = factor

    def __call__(self, img):
        h, w = img.shape[:2]
        small = cv2.resize(img, (w//self.factor, h//self.factor), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), "Aliasing"
