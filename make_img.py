import os
import numpy as np
from skimage.util import random_noise
from skimage.draw import disk, rectangle
from skimage import img_as_ubyte
from skimage.io import imsave

# Settings
output_dir = "dataset"  # Thay đổi đường dẫn thành thư mục trong dự án
os.makedirs(f"{output_dir}/circle", exist_ok=True)
os.makedirs(f"{output_dir}/square", exist_ok=True)

image_size = 64
radius = 20
square_size = 40
num_images = 100

# Generate 100 circle and 100 square images with salt & pepper noise
for i in range(num_images):
    # Circle
    circle = np.zeros((image_size, image_size))
    rr, cc = disk((image_size // 2, image_size // 2), radius)
    circle[rr, cc] = 1
    noisy_circle = random_noise(circle, mode='s&p', amount=0.05)
    circle_uint8 = img_as_ubyte(noisy_circle)
    imsave(f"{output_dir}/circle/circle_{i:03d}.png", circle_uint8)

    # Square
    square = np.zeros((image_size, image_size))
    top_left = ((image_size - square_size) // 2, (image_size - square_size) // 2)
    bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
    rr, cc = rectangle(start=top_left, end=bottom_right, shape=square.shape)
    square[rr, cc] = 1
    noisy_square = random_noise(square, mode='s&p', amount=0.05)
    square_uint8 = img_as_ubyte(noisy_square)
    imsave(f"{output_dir}/square/square_{i:03d}.png", square_uint8)

print(f"Generated {num_images} circle images and {num_images} square images in {output_dir}")
