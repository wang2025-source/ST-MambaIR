import os
import cv2
import argparse
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_image(path):
    """read image and normalize to [0, 1] float32 RGB"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def calculate_metrics(img1, img2):
    """calculate PSNR and SSIM of a single image"""
    psnr = peak_signal_noise_ratio(img1, img2, data_range=1.0)
    ssim = structural_similarity(img1, img2, win_size=3, channel_axis=-1, data_range=1.0)
    return psnr, ssim

def batch_eval(gt_dir, pred_dir):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    assert len(gt_files) == len(pred_files), "different number of images in gt_dir and pred_dir"

    total_psnr, total_ssim = 0, 0
    num_files = len(gt_files)

    for i in range(num_files):
        gt_path = os.path.join(gt_dir, gt_files[i])
        pred_path = os.path.join(pred_dir, pred_files[i])
        gt_img = load_image(gt_path)
        pred_img = load_image(pred_path)

        if gt_img.shape != pred_img.shape:
            print(f"[skip] different image size: {gt_files[i]}")
            continue

        psnr, ssim = calculate_metrics(gt_img, pred_img)
        total_psnr += psnr
        total_ssim += ssim

        print(f"[{i+1}/{num_files}] {gt_files[i]} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

    avg_psnr = total_psnr / num_files
    avg_ssim = total_ssim / num_files
    print("\n=== evaluation finished ===")
    print(f"mean PSNR: {avg_psnr:.2f} dB")
    print(f"mean SSIM: {avg_ssim:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch PSNR & SSIM Evaluation')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT image dir')
    parser.add_argument('--pred_dir', type=str, required=True, help='predicted image dir')
    args = parser.parse_args()

    batch_eval(args.gt_dir, args.pred_dir)
