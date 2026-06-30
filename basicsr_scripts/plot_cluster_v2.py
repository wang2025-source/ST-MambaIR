import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from basicsr.archs.mambairv2_arch import MambaIRv2 # 确保路径正确

def plot_multipole_clusters(model, img, stage_idx=2, block_idx=0):
    """
    model: instantiated model object
    img: original infrared image (numpy array)
    stage_idx: index of depths
    block_idx: index of block
    """
    # 1. extract cluster data
    # model -> layers[stage] -> residual_group -> layers[block] -> assm
    try:
        target_assm = model.layers[stage_idx].residual_group.layers[block_idx].assm
        cluster_id_map = target_assm.cluster_ids # 形状 [1, n]
    except AttributeError:
        print(f"cluster_ids not found at stage {stage_idx}, block {block_idx}, check whether saved and correct the index")
        return

    # 2. reshape to 2D and draw
    # h, w should equal to feature image size input to ASSM
    h_orig, w_orig = img.shape
    res_map = cluster_id_map[0].view(h_orig, w_orig).cpu().numpy()

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Infrared Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(res_map, cmap='nipy_spectral')
    plt.title(f"Multipole Clusters (Stage {stage_idx}, Block {block_idx})")
    plt.axis('off')

    plt.tight_layout()
    save_name = f"vis_stage{stage_idx}_block{block_idx}.png"
    plt.savefig(save_name, dpi=300)
    print(f"visualization saved to {save_name}")
    plt.close()

# ================= use case =================
if __name__ == '__main__':
    # same as training settings
    my_model = MambaIRv2(
        upscale=1,
        in_chans=3,
        img_size=64,
        img_range=1.,
        # embed_dim=32, # param 2.1M
        embed_dim= 64, # param 6.9M
        # embed_dim= 48 # param 4.2M
        d_state=16,
        depths=[4, 4, 4,4,4,4],
        num_heads=[4,4,4,4,4,4],
        window_size=16,
        inner_rank=64,
        num_tokens=32,
        convffn_kernel_size=5,
        mlp_ratio=2.
    )

    checkpoint_path = 'experiments/HM-TIR-MULTIPOLE/wavelet-focalnet_multipole-mamba_K32_3e-4_67epoch_6.9M_0108/models/net_g_20000.pth'
    img_path='multi_noise_addition/out_noisy_image/00000.png'

    # 1. load weights once
    print(f"loading weights: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    if 'params' in checkpoint:
        my_model.load_state_dict(checkpoint['params'])
    else:
        my_model.load_state_dict(checkpoint)
    my_model.eval().cuda()

    # 2. read infrared image once
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError("invalid image path")
    
    # normalize and transfer to Tensor [1, 3, H, W]
    img_input = img.astype(np.float32) / 255.
    img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda()

    # 3. inference once (save all cluster_ids)
    print("Running inference...")
    with torch.no_grad():
        # model.forward deal with padding and params
        _ = my_model(img_tensor)

    for i in range(6):
        for j in range(4):
            plot_multipole_clusters(
                model=my_model,
                img=img,
                stage_idx=i,
                block_idx=j
            )
