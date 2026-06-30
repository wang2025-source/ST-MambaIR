#!/usr/bin/env python
"""
批量验证脚本 - 对所有模型在所有数据集上进行验证
"""
import os
import yaml
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


# 数据集配置
DATASETS = {
    'hm_tir': {
        'gt': '/home/blu/work/proj/infrared-denoise-working/datasets/hm_tir/val/gt',
        'lq': '/home/blu/work/proj/infrared-denoise-working/datasets/hm_tir/val/noise',
    },
    'INFRARED_CVC09': {
        'gt': '/home/blu/work/proj/infrared-denoise-working/datasets/INFRARED_CVC09/val/gt',
        'lq': '/home/blu/work/proj/infrared-denoise-working/datasets/INFRARED_CVC09/val/noise',
    },
    'Rivadeneira2020': {
        'gt': '/home/blu/work/proj/infrared-denoise-working/datasets/Rivadeneira2020/val/gt',
        'lq': '/home/blu/work/proj/infrared-denoise-working/datasets/Rivadeneira2020/val/noise',
    },
}

# 模型权重目录
WEIGHTS_DIR = '/home/blu/playground/weights'


def find_weight_file(model_dir):
    """查找模型权重文件（包括子目录）"""
    # 先在当前目录查找
    for f in os.listdir(model_dir):
        if f.startswith('net_g') and f.endswith('.pth'):
            return os.path.join(model_dir, f)
    # 再在子目录查找
    for sub in os.listdir(model_dir):
        sub_path = os.path.join(model_dir, sub)
        if os.path.isdir(sub_path):
            for f in os.listdir(sub_path):
                if f.startswith('net_g') and f.endswith('.pth'):
                    return os.path.join(sub_path, f)
    return None


def find_config_file(model_dir):
    """查找模型配置文件（包括子目录）"""
    # 先在当前目录查找
    for f in os.listdir(model_dir):
        if f.endswith('.yml') or f.endswith('.yaml'):
            return os.path.join(model_dir, f)
    # 再在子目录查找
    for sub in os.listdir(model_dir):
        sub_path = os.path.join(model_dir, sub)
        if os.path.isdir(sub_path):
            for f in os.listdir(sub_path):
                if f.endswith('.yml') or f.endswith('.yaml'):
                    return os.path.join(sub_path, f)
    return None

def get_model_info(config_path):
    """从配置文件中提取模型信息"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    network_g = config.get('network_g', {})
    model_type = config.get('model_type', 'ImageRestorationModel')
    
    return {
        'network_g': network_g,
        'model_type': model_type,
    }


def create_test_config(model_name, model_dir, dataset_name, dataset_info, output_dir, save_img=True):
    """创建测试配置文件"""
    config_path = find_config_file(model_dir)
    weight_path = find_weight_file(model_dir)
    
    if not config_path or not weight_path:
        return None
    
    model_info = get_model_info(config_path)
    
    test_config = {
        'name': f'{model_name}_{dataset_name}_test',
        'model_type': model_info['model_type'],
        'scale': 1,
        'num_gpu': 1,
        'manual_seed': 10,
        'datasets': {
            'test_1': {
                'name': f'{dataset_name}_val',
                'type': 'PairedImageDataset',
                'dataroot_gt': dataset_info['gt'],
                'dataroot_lq': dataset_info['lq'],
                'io_backend': {'type': 'disk'},
            }
        },
        'network_g': model_info['network_g'],
        'path': {
            'pretrain_network_g': weight_path,
            'strict_load_g': True,
        },
        'val': {
            'save_img': save_img,
            'suffix': None,
            'metrics': {
                'psnr': {
                    'type': 'calculate_psnr',
                    'crop_border': 0,
                    'test_y_channel': False,
                },
                'ssim': {
                    'type': 'calculate_ssim',
                    'crop_border': 0,
                    'test_y_channel': False,
                }
            }
        }
    }
    
    # 添加 window_size 对于 Restormer/Xformer 等模型
    if model_info['network_g'].get('type') in ['Restormer', 'Xformer', 'IDTransformer']:
        test_config['val']['window_size'] = 8
        test_config['val']['rgb2bgr'] = True
        test_config['val']['max_minibatch'] = 8
    
    config_filename = f'test_{model_name}_{dataset_name}.yml'
    config_path_out = os.path.join(output_dir, config_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path_out, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    return config_path_out


def run_test(config_path, log_dir):
    """运行测试"""
    log_file = os.path.join(log_dir, f"{os.path.basename(config_path).replace('.yml', '.log')}")
    
    cmd = [
        'conda', 'run', '-n', 'nafmamba',
        'python', '-m', 'basicsr.test',
        '-opt', config_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='批量验证脚本')
    parser.add_argument('--output_dir', type=str, default='options/test/batch_validation',
                        help='测试配置输出目录')
    parser.add_argument('--log_dir', type=str, default='results/batch_validation_logs',
                        help='日志输出目录')
    parser.add_argument('--save_img', action='store_true', default=True,
                        help='是否保存输出图像')
    parser.add_argument('--dry_run', action='store_true',
                        help='只生成配置文件，不运行测试')
    parser.add_argument('--models', type=str, nargs='*', default=None,
                        help='指定要测试的模型名称（可选）')
    parser.add_argument('--datasets', type=str, nargs='*', default=None,
                        help='指定要测试的数据集名称（可选）')
    
    args = parser.parse_args()
    
    # 获取所有模型
    models = []
    for iter_type in ['200k', '20k']:
        iter_dir = os.path.join(WEIGHTS_DIR, iter_type)
        if not os.path.exists(iter_dir):
            continue
        for model_dir_name in os.listdir(iter_dir):
            model_dir = os.path.join(iter_dir, model_dir_name)
            if os.path.isdir(model_dir):
                models.append({
                    'name': f'{model_dir_name}',
                    'dir': model_dir,
                    'iter': iter_type,
                })
    
    # 过滤模型
    if args.models:
        models = [m for m in models if any(model_name in m['name'] for model_name in args.models)]
    
    # 过滤数据集
    datasets = DATASETS
    if args.datasets:
        datasets = {k: v for k, v in DATASETS.items() if k in args.datasets}
    
    print(f"Found {len(models)} models and {len(datasets)} datasets")
    print(f"Total test tasks: {len(models) * len(datasets)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 生成测试配置并运行
    results = []
    for model in models:
        for dataset_name, dataset_info in datasets.items():
            print(f"\n{'='*60}")
            print(f"Model: {model['name']}, Dataset: {dataset_name}")
            print(f"{'='*60}")
            
            config_path = create_test_config(
                model['name'], model['dir'], dataset_name, dataset_info,
                args.output_dir, args.save_img
            )
            
            if not config_path:
                print(f"  Skipping: No config or weight file found")
                results.append({
                    'model': model['name'],
                    'dataset': dataset_name,
                    'status': 'skipped',
                    'reason': 'No config or weight file'
                })
                continue
            
            print(f"  Config: {config_path}")
            
            if args.dry_run:
                print(f"  [DRY RUN] Would run test")
                results.append({
                    'model': model['name'],
                    'dataset': dataset_name,
                    'status': 'dry_run',
                    'config': config_path,
                })
                continue
            
            return_code = run_test(config_path, args.log_dir)
            
            results.append({
                'model': model['name'],
                'dataset': dataset_name,
                'status': 'success' if return_code == 0 else 'failed',
                'config': config_path,
                'return_code': return_code,
            })
            
            print(f"  Status: {'SUCCESS' if return_code == 0 else 'FAILED'}")
    
    # 打印汇总
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    
    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    
    # 保存结果到文件
    results_file = os.path.join(args.log_dir, f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()