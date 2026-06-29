import os
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def main(args):
    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in sorted(input_dir.iterdir()) if is_image_file(f.name)]

    for image_path in tqdm(image_files, desc="Denoising images"):
        output_image_path = output_dir / image_path.name

        command = [
            "python", "basicsr/demo.py",
            "-opt", args.config,
            "--input_path", str(image_path),
            "--output_path", str(output_image_path)
        ]

        subprocess.run(command, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch image denoising using basicsr/demo.py")
    parser.add_argument('--input_path', type=str, required=True, help='Path to directory containing noisy images')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save denoised images')
    parser.add_argument('--config', type=str, required=True, help='Path to NAFNet config yml file')

    args = parser.parse_args()
    main(args)
