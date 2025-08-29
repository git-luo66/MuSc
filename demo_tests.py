#!/usr/bin/env python3
"""
MuSc测试演示脚本
快速演示测试功能
"""
import sys
import os

# 添加项目路径
sys.path.append(os.getcwd())

def demo_unit_tests():
    """演示单元测试"""
    print("🧪 演示单元测试...")
    
    # 演示metrics测试
    from utils.metrics import compute_metrics
    import numpy as np
    
    # 生成测试数据
    gt_sp = np.array([0, 0, 1, 1])
    pr_sp = np.array([0.1, 0.2, 0.8, 0.9])
    
    image_metrics, pixel_metrics = compute_metrics(gt_sp=gt_sp, pr_sp=pr_sp)
    
    print(f"  ✅ 图像级指标: AUROC={image_metrics[0]:.3f}, F1={image_metrics[1]:.3f}, AP={image_metrics[2]:.3f}")
    print(f"  ✅ 像素级指标: AUROC={pixel_metrics[0]:.3f}, F1={pixel_metrics[1]:.3f}, AP={pixel_metrics[2]:.3f}, AUPRO={pixel_metrics[3]:.3f}")


def demo_config_loading():
    """演示配置加载"""
    print("📁 演示配置加载...")
    
    from utils.load_config import load_yaml
    
    try:
        config = load_yaml('configs/musc.yaml')
        print(f"  ✅ 成功加载配置文件")
        print(f"  📋 数据集: {config['datasets']['dataset_name']}")
        print(f"  🏗️  骨干网络: {config['models']['backbone_name']}")
        print(f"  🎯 设备: {config['device']}")
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")


def demo_module_tests():
    """演示模块测试"""
    print("🔧 演示模块测试...")
    
    import torch
    
    # 测试MSM模块
    try:
        from models.modules._MSM import compute_scores_fast
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Z = torch.randn(3, 5, 32).to(device)
        
        scores = compute_scores_fast(Z, 1, device)
        print(f"  ✅ MSM模块测试通过，输出形状: {scores.shape}")
        
    except Exception as e:
        print(f"  ❌ MSM模块测试失败: {e}")
    
    # 测试RsCIN模块
    try:
        from models.modules._RsCIN import RsCIN
        
        scores_old = np.array([0.2, 0.5, 0.8])
        cls_tokens = np.random.randn(3, 32)
        
        result = RsCIN(scores_old, cls_tokens, [1])
        print(f"  ✅ RsCIN模块测试通过，输出形状: {result.shape}")
        
    except Exception as e:
        print(f"  ❌ RsCIN模块测试失败: {e}")


def demo_mock_data():
    """演示模拟数据生成"""
    print("🎭 演示模拟数据生成...")
    
    try:
        from tests.data.mock_data import TestDataGenerator, MockBackbone
        
        # 生成特征数据
        features = TestDataGenerator.generate_feature_tensors(batch_size=2, num_layers=2)
        print(f"  ✅ 生成特征数据: {len(features)} 层，形状 {features[0].shape}")
        
        # 生成异常检测数据
        images, labels, masks, paths = TestDataGenerator.generate_anomaly_data(num_samples=5)
        print(f"  ✅ 生成异常检测数据: {len(images)} 个样本")
        
        # 测试模拟骨干网络
        mock_backbone = MockBackbone()
        test_images = torch.randn(2, 3, 224, 224)
        mock_features = mock_backbone(test_images)
        print(f"  ✅ 模拟骨干网络输出: {len(mock_features)} 层特征")
        
    except Exception as e:
        print(f"  ❌ 模拟数据生成失败: {e}")


def main():
    """主演示函数"""
    print("🎉 欢迎使用MuSc测试套件演示！")
    print("=" * 50)
    
    # 检查环境
    print(f"🐍 Python版本: {sys.version}")
    print(f"📍 工作目录: {os.getcwd()}")
    print(f"🔥 PyTorch版本: {torch.__version__ if 'torch' in sys.modules else '未安装'}")
    print("")
    
    # 运行演示
    demo_config_loading()
    print("")
    
    demo_unit_tests()
    print("")
    
    demo_module_tests()
    print("")
    
    demo_mock_data()
    print("")
    
    print("🎊 演示完成！")
    print("")
    print("💡 要运行完整测试套件，请使用:")
    print("   python run_tests.py --all")
    print("   或者")
    print("   make test")
    print("")
    print("📚 查看测试文档: tests/README.md")


if __name__ == "__main__":
    main()