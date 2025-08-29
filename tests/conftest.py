"""
测试配置文件
提供测试fixtures和共用的测试工具
"""
import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
import yaml
from unittest.mock import MagicMock


@pytest.fixture
def device():
    """提供测试设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_config():
    """提供测试用的配置"""
    return {
        'datasets': {
            'dataset_name': 'mvtec_ad',
            'data_path': './test_data/',
            'class_name': 'bottle',
            'img_resize': 224,
            'divide_num': 1
        },
        'models': {
            'backbone_name': 'ViT-B-16',
            'pretrained': 'openai',
            'batch_size': 2,
            'feature_layers': [5, 11],
            'r_list': [1, 3]
        },
        'device': '0',
        'testing': {
            'output_dir': './test_output',
            'vis': False,
            'vis_type': 'single_norm',
            'save_excel': False
        }
    }


@pytest.fixture
def temp_config_file(sample_config):
    """创建临时配置文件"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(sample_config, temp_file)
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name)


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image_data():
    """生成测试用的图像数据"""
    return torch.randn(4, 3, 224, 224)  # batch_size=4, RGB, 224x224


@pytest.fixture
def sample_mask_data():
    """生成测试用的掩码数据"""
    return torch.randint(0, 2, (4, 224, 224))  # 二值掩码


@pytest.fixture
def sample_features():
    """生成测试用的特征数据"""
    return torch.randn(4, 196, 768)  # batch_size=4, 196 patches, 768 features


@pytest.fixture
def sample_anomaly_scores():
    """生成测试用的异常分数"""
    image_scores = torch.rand(4)  # 图像级别分数
    pixel_scores = torch.rand(4, 224, 224)  # 像素级别分数
    return image_scores, pixel_scores


@pytest.fixture
def mock_dataset():
    """创建模拟数据集"""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    dataset.__getitem__ = MagicMock(return_value=(
        torch.randn(3, 224, 224),  # image
        torch.randint(0, 2, (224, 224)),  # mask
        0,  # label (0=normal, 1=anomaly)
        "test_image.png"  # image_path
    ))
    return dataset


@pytest.fixture(autouse=True)
def setup_test_environment():
    """自动设置测试环境"""
    # 设置随机种子确保测试可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 禁用警告
    import warnings
    warnings.filterwarnings("ignore")
    
    yield
    
    # 清理测试环境
    if torch.cuda.is_available():
        torch.cuda.empty_cache()