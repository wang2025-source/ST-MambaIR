import os
import torch
import yaml
from basicsr.archs import build_network
from thop import profile, clever_format


# Patch mamba_ssm's selective_scan_cuda to avoid stride(-1) == 1 error for sequence length 1
try:
    import torch
    import selective_scan_cuda
    original_fwd = selective_scan_cuda.fwd

    def patched_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        def fix_stride(t):
            if t is not None and t.stride(-1) != 1:
                return torch.empty_like(t, memory_format=torch.contiguous_format).copy_(t)
            return t
        
        u = fix_stride(u)
        delta = fix_stride(delta)
        A = fix_stride(A)
        B = fix_stride(B)
        C = fix_stride(C)
        D = fix_stride(D)
        z = fix_stride(z)
        delta_bias = fix_stride(delta_bias)
        return original_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

    selective_scan_cuda.fwd = patched_fwd
except ImportError:
    pass

def get_flops_and_params(yaml_path, input_size=(1, 3, 256, 256)):
    with open(yaml_path, 'r') as f:
        opt = yaml.safe_load(f)
    
    if 'network_g' not in opt:
        return None, None, f"No 'network_g' in {yaml_path}"
    
    try:
        net = build_network(opt['network_g'])
        net.eval()
        input_data = torch.randn(*input_size)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        input_data = input_data.to(device)
        
        flops, params = profile(net, inputs=(input_data,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops_str, params_str, None
    except Exception as e:
        return None, None, str(e)

def main():
    custom_dir = 'options/train/CUSTOM/'
    yaml_files = sorted([f for f in os.listdir(custom_dir) if f.endswith('.yml')])
    
    print(f"{'YAML File':<30} | {'Type':<15} | {'FLOPs':<10} | {'Params':<10}")
    print("-" * 75)
    
    for yaml_file in yaml_files:
        yaml_path = os.path.join(custom_dir, yaml_file)
        with open(yaml_path, 'r') as f:
            opt = yaml.safe_load(f)
        
        net_type = opt.get('network_g', {}).get('type', 'Unknown')
        flops, params, error = get_flops_and_params(yaml_path)
        
        if error:
            print(f"{yaml_file:<30} | {net_type:<15} | Error: {error}")
        else:
            print(f"{yaml_file:<30} | {net_type:<15} | {flops:<10} | {params:<10}")

if __name__ == '__main__':
    main()
