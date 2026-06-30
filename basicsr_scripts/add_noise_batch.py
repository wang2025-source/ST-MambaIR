import os
import argparse
import cv2
import numpy as np
import torch
import random
from tqdm import tqdm

def add_noise(img_tensor, noiseIntL=[0.01, 0.1]):
    """
    add non-uniform noise to a image, keep the size of the image, and the
    noise of each channel is the same.
    input:
        img_tensor: [C, H, W] in [0, 1]
    output:
        noisy_img_tensor: [C, H, W] in [0, 1]
    """
    C, H, W = img_tensor.shape
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    noise_S = torch.zeros_like(img_tensor)

    # same noise for each channel
    beta1 = np.random.uniform(noiseIntL[0], noiseIntL[1])
    beta2 = np.random.uniform(noiseIntL[0], noiseIntL[1])
    beta3 = np.random.uniform(noiseIntL[0], noiseIntL[1])
    beta4 = np.random.uniform(noiseIntL[0], noiseIntL[1])

    # generate a noise template
    A1 = np.random.normal(0, beta1, size=W)
    A2 = np.random.normal(0, beta2, size=W)
    A3 = np.random.normal(0, beta3, size=W)
    A4 = np.random.normal(0, beta4, size=W)

    A1 = np.tile(A1, (H, 1))  # 复制成 [H, W]
    A2 = np.tile(A2, (H, 1))
    A3 = np.tile(A3, (H, 1))
    A4 = np.tile(A4, (H, 1))

    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    A3 = torch.from_numpy(A3).float()
    A4 = torch.from_numpy(A4).float()

    # each channel share the same noise
    for c in range(C):
        img = img_tensor[0, c]
        noisy = A1 + A2 * img + A3 * A3 * img + A4 * A4 * A4 * img + img
        noise_S[0, c] = torch.clip(noisy, 0., 1.)

    return noise_S.squeeze(0)  # [C, H, W]


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
    return img_tensor


def save_image(tensor, path):
    img = tensor.permute(1, 2, 0).numpy()  # [H, W, C]
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    img_names = [f for f in os.listdir(args.in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for name in tqdm(img_names, desc="Adding noise"):
        img_path = os.path.join(args.in_dir, name)
        out_path = os.path.join(args.out_dir, name)

        img_tensor = load_image(img_path)
        noisy_tensor = add_noise(img_tensor, noiseIntL=args.noise_range)

        save_image(noisy_tensor, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate noisy images using custom noise model.")
    parser.add_argument('--in_dir', type=str, required=True, help='Directory of clean input images')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save noisy images')
    parser.add_argument('--noise_range', nargs=2, type=float, default=[0.05, 0.15], help='Noise intensity range')

    args = parser.parse_args()
    main(args)
