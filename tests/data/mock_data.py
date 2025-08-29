"""
测试数据生成和Mock对象
为测试提供模拟数据和对象
"""
import torch
import numpy as np
import os
import tempfile
from PIL import Image
from unittest.mock import MagicMock


class MockDatasetGenerator:
    """模拟数据集生成器"""
    
    @staticmethod
    def create_mock_mvtec_structure(base_path, category='bottle', num_normal=5, num_anomaly=3):
        """创建模拟MVTec AD数据集结构"""
        # 创建目录结构
        test_normal_dir = os.path.join(base_path, category, 'test', 'good')
        test_anomaly_dir = os.path.join(base_path, category, 'test', 'broken_large')
        ground_truth_dir = os.path.join(base_path, category, 'ground_truth', 'broken_large')
        
        os.makedirs(test_normal_dir, exist_ok=True)
        os.makedirs(test_anomaly_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)
        
        # 生成正常图像
        for i in range(num_normal):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(test_normal_dir, f'normal_{i:03d}.png'))
        
        # 生成异常图像和对应的ground truth
        for i in range(num_anomaly):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(test_anomaly_dir, f'anomaly_{i:03d}.png'))
            
            # 生成ground truth掩码
            mask = np.random.randint(0, 2, (224, 224), dtype=np.uint8) * 255
            mask_img = Image.fromarray(mask)
            mask_img.save(os.path.join(ground_truth_dir, f'anomaly_{i:03d}_mask.png'))
    
    @staticmethod
    def create_mock_visa_structure(base_path, category='candle', num_normal=5, num_anomaly=3):
        """创建模拟VisA数据集结构"""
        test_normal_dir = os.path.join(base_path, category, 'Data', 'Images', 'Normal')
        test_anomaly_dir = os.path.join(base_path, category, 'Data', 'Images', 'Anomaly')
        ground_truth_dir = os.path.join(base_path, category, 'Data', 'Masks', 'Anomaly')
        
        os.makedirs(test_normal_dir, exist_ok=True)
        os.makedirs(test_anomaly_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)
        
        # 生成图像和掩码（与MVTec类似）
        for i in range(num_normal):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(test_normal_dir, f'normal_{i:04d}.JPG'))
        
        for i in range(num_anomaly):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(test_anomaly_dir, f'anomaly_{i:04d}.JPG'))
            
            mask = np.random.randint(0, 2, (224, 224), dtype=np.uint8) * 255
            mask_img = Image.fromarray(mask)
            mask_img.save(os.path.join(ground_truth_dir, f'anomaly_{i:04d}.png'))


class MockBackbone:
    """模拟骨干网络"""
    
    def __init__(self, feature_dim=768, num_layers=24):
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.eval_mode = True
    
    def __call__(self, images):
        """模拟骨干网络前向传播"""
        batch_size = images.shape[0]
        # 模拟ViT输出：每层都有197个token（196 patches + 1 CLS）
        features = []
        for _ in range(self.num_layers):
            layer_features = torch.randn(batch_size, 197, self.feature_dim)
            features.append(layer_features)
        return features
    
    def eval(self):
        """设置为评估模式"""
        self.eval_mode = True
        return self
    
    def to(self, device):
        """移动到指定设备"""
        return self
    
    def parameters(self):
        """返回模拟参数"""
        return []


class MockPreprocess:
    """模拟预处理器"""
    
    def __call__(self, image):
        """模拟图像预处理"""
        if isinstance(image, Image.Image):
            # 转换PIL图像到tensor
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
            tensor = torch.from_numpy(img_array).float() / 255.0
            return tensor
        return image


class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_feature_tensors(batch_size=4, num_patches=196, feature_dim=768, num_layers=4):
        """生成特征张量"""
        features = []
        for _ in range(num_layers):
            # 添加CLS token
            layer_feature = torch.randn(batch_size, num_patches + 1, feature_dim)
            features.append(layer_feature)
        return features
    
    @staticmethod
    def generate_anomaly_data(num_samples=10, image_size=224):
        """生成异常检测测试数据"""
        # 生成图像数据
        images = torch.randn(num_samples, 3, image_size, image_size)
        
        # 生成标签（0=正常，1=异常）
        labels = torch.randint(0, 2, (num_samples,))
        
        # 生成像素级掩码
        masks = torch.randint(0, 2, (num_samples, image_size, image_size))
        
        # 生成图像路径
        paths = [f'test_image_{i:03d}.png' for i in range(num_samples)]
        
        return images, labels, masks, paths
    
    @staticmethod
    def generate_similarity_matrix(num_images=10, feature_dim=64):
        """生成相似度矩阵测试数据"""
        # 生成随机特征
        features = np.random.randn(num_images, feature_dim)
        
        # 计算相似度矩阵
        similarity_matrix = features @ features.T
        
        # 归一化到[0,1]
        similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (
            similarity_matrix.max() - similarity_matrix.min()
        )
        
        # 确保对角线为1
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return features, similarity_matrix


def create_test_config(temp_dir, dataset_name='mvtec_ad', category='bottle'):
    """创建测试配置"""
    return {
        'datasets': {
            'dataset_name': dataset_name,
            'data_path': temp_dir,
            'class_name': category,
            'img_resize': 224,
            'divide_num': 1
        },
        'models': {
            'backbone_name': 'ViT-B-16',
            'pretrained': 'openai',
            'batch_size': 2,
            'feature_layers': [5, 11, 17, 23],
            'r_list': [1, 3, 5]
        },
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'testing': {
            'output_dir': os.path.join(temp_dir, 'output'),
            'vis': False,
            'vis_type': 'single_norm',
            'save_excel': False
        }
    }


def setup_mock_environment():
    """设置模拟环境"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 生成模拟数据集
    MockDatasetGenerator.create_mock_mvtec_structure(temp_dir)
    
    return temp_dir


def teardown_mock_environment(temp_dir):
    """清理模拟环境"""
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)