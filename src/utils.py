import numpy as np
from PIL import Image


def transform(image):
    return image / 127.5 - 1.


def inverse_transform(image):
    return (image + 1.) / 2


def save_image(image, save_path):
    img = np.squeeze(image, axis=0)
    img = Image.fromarray((((img + 1.0) / 2) * 255.0).astype('uint8'))
    img.save(save_path)
