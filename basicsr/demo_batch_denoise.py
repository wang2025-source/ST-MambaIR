import os
import torch

from glob import glob
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, tensor2img, imwrite

def main():
    # 1. Load config
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()
    opt['dist'] = False

    input_folder = opt['img_path'].get('input_img')
    output_folder = opt['img_path'].get('output_img')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_client = FileClient('disk')
    model = create_model(opt)

    # 2. Loop through all PNG images in folder
    image_list = sorted(glob(os.path.join(input_folder, '*.png')))
    if not image_list:
        raise FileNotFoundError(f"No PNG images found in {input_folder}")

    for img_path in image_list:
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_folder, img_name)

        # 3. Read image
        img_bytes = file_client.get(img_path, None)
        if img_bytes is None:
            print(f"[Warning] Cannot read {img_path}. Skipping.")
            continue
        try:
            img = imfrombytes(img_bytes, float32=True)
        except:
            print(f"[Error] Failed to decode image: {img_path}")
            continue

        img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0)

        # 4. Inference
        model.feed_data(data={'lq': img})

        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        # 5. Save result
        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, output_path)
        print(f"[OK] Denoised: {img_name}")

if __name__ == '__main__':
    main()
