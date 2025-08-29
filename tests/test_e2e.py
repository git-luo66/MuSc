"""
端到端测试
测试完整的MuSc工作流程
"""
import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
import yaml
from unittest.mock import patch, MagicMock
from tests.data.mock_data import (
    MockDatasetGenerator, MockBackbone, MockPreprocess,
    create_test_config, setup_mock_environment, teardown_mock_environment
)


class TestMuScE2E:
    """MuSc端到端测试"""
    
    @pytest.fixture
    def e2e_temp_dir(self):
        """端到端测试临时目录"""
        temp_dir = setup_mock_environment()
        yield temp_dir
        teardown_mock_environment(temp_dir)
    
    @pytest.fixture
    def e2e_config_file(self, e2e_temp_dir):
        """端到端测试配置文件"""
        config = create_test_config(e2e_temp_dir)
        config_path = os.path.join(e2e_temp_dir, 'test_config.yaml')
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @patch('models.musc._backbones.load')
    def test_e2e_config_loading(self, mock_backbone_load, e2e_config_file):
        """测试端到端配置加载"""
        mock_backbone_load.return_value = (MockBackbone(), MockPreprocess())
        
        from utils.load_config import load_yaml
        from models.musc import MuSc
        
        # 加载配置
        config = load_yaml(os.path.basename(e2e_config_file))
        
        # 创建模型
        model = MuSc(config, seed=42)
        
        assert model.dataset == 'mvtec_ad'
        assert model.categories == 'bottle'
    
    @patch('models.musc._backbones.load')
    @patch('models.musc.mvtec.MVTecDataset')
    def test_e2e_data_loading(self, mock_dataset, mock_backbone_load, e2e_temp_dir):
        """测试端到端数据加载"""
        mock_backbone_load.return_value = (MockBackbone(), MockPreprocess())
        
        # 模拟数据集
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__ = MagicMock(return_value=8)
        mock_dataset_instance.__getitem__ = MagicMock(side_effect=lambda idx: (
            torch.randn(3, 224, 224),
            torch.randint(0, 2, (224, 224)),
            idx % 2,
            f"test_{idx}.png"
        ))
        mock_dataset.return_value = mock_dataset_instance
        
        config = create_test_config(e2e_temp_dir)
        model = MuSc(config, seed=42)
        
        # 测试数据集加载
        dataset = model.load_datasets('bottle')
        assert dataset is not None
        assert len(dataset) == 8
    
    @pytest.mark.slow
    @patch('models.musc._backbones.load')
    def test_e2e_feature_extraction_pipeline(self, mock_backbone_load, e2e_temp_dir):
        """测试端到端特征提取流水线"""
        # 设置模拟骨干网络
        mock_backbone = MockBackbone(feature_dim=768, num_layers=24)
        mock_preprocess = MockPreprocess()
        mock_backbone_load.return_value = (mock_backbone, mock_preprocess)
        
        config = create_test_config(e2e_temp_dir)
        config['models']['batch_size'] = 1  # 减小批次大小加速测试
        
        model = MuSc(config, seed=42)
        
        # 验证模型组件初始化
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'preprocess')
        assert model.feature_layers == [5, 11, 17, 23]
        assert model.r_list == [1, 3, 5]
    
    def test_e2e_metrics_calculation(self):
        """测试端到端指标计算"""
        from utils.metrics import compute_metrics
        
        # 生成测试数据
        gt_sp = np.array([0, 0, 1, 1, 0, 1])
        pr_sp = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        gt_px = np.random.randint(0, 2, (6, 50, 50))
        pr_px = np.random.rand(6, 50, 50)
        
        image_metrics, pixel_metrics = compute_metrics(gt_sp, pr_sp, gt_px, pr_px)
        
        # 验证指标计算结果
        assert len(image_metrics) == 3  # AUROC, F1, AP
        assert len(pixel_metrics) == 4   # AUROC, F1, AP, AUPRO
        
        # 所有指标应该在合理范围内
        for metric in image_metrics + pixel_metrics:
            assert 0 <= metric <= 1
    
    def test_e2e_output_directory_structure(self, e2e_temp_dir):
        """测试端到端输出目录结构"""
        config = create_test_config(e2e_temp_dir)
        
        from models.musc import MuSc
        with patch('models.musc._backbones.load') as mock_load:
            mock_load.return_value = (MockBackbone(), MockPreprocess())
            model = MuSc(config, seed=42)
        
        # 验证输出目录被创建
        expected_output_dir = os.path.join(e2e_temp_dir, 'output')
        assert os.path.exists(expected_output_dir)
    
    def test_e2e_configuration_override(self, e2e_temp_dir):
        """测试端到端配置覆盖"""
        config = create_test_config(e2e_temp_dir)
        
        # 测试配置参数覆盖
        config['models']['batch_size'] = 8
        config['datasets']['img_resize'] = 512
        
        with patch('models.musc._backbones.load') as mock_load:
            mock_load.return_value = (MockBackbone(), MockPreprocess())
            model = MuSc(config, seed=42)
        
        assert model.batch_size == 8
        assert model.image_size == 512
    
    @pytest.mark.integration
    def test_e2e_error_handling(self, e2e_temp_dir):
        """测试端到端错误处理"""
        config = create_test_config(e2e_temp_dir)
        
        # 测试无效数据路径
        config['datasets']['data_path'] = '/nonexistent/path'
        
        with patch('models.musc._backbones.load') as mock_load:
            mock_load.return_value = (MockBackbone(), MockPreprocess())
            
            # 应该在数据路径验证时失败
            with pytest.raises(AssertionError):
                model = MuSc(config, seed=42)
    
    def test_e2e_different_datasets(self, e2e_temp_dir):
        """测试不同数据集的端到端流程"""
        datasets = ['mvtec_ad', 'visa', 'btad']
        categories = ['bottle', 'candle', '01']
        
        for dataset, category in zip(datasets, categories):
            config = create_test_config(e2e_temp_dir, dataset, category)
            
            # 为每个数据集创建相应的目录结构
            if dataset == 'mvtec_ad':
                MockDatasetGenerator.create_mock_mvtec_structure(e2e_temp_dir, category)
            elif dataset == 'visa':
                MockDatasetGenerator.create_mock_visa_structure(e2e_temp_dir, category)
            
            with patch('models.musc._backbones.load') as mock_load:
                mock_load.return_value = (MockBackbone(), MockPreprocess())
                
                try:
                    model = MuSc(config, seed=42)
                    assert model.dataset == dataset
                    assert model.categories == category
                except Exception as e:
                    pytest.fail(f"数据集 {dataset} 初始化失败: {e}")
    
    @pytest.mark.slow
    def test_e2e_memory_management(self, e2e_temp_dir):
        """测试端到端内存管理"""
        if not torch.cuda.is_available():
            pytest.skip("需要CUDA进行内存测试")
        
        config = create_test_config(e2e_temp_dir)
        config['device'] = '0'
        config['models']['batch_size'] = 4
        
        with patch('models.musc._backbones.load') as mock_load:
            mock_load.return_value = (MockBackbone(), MockPreprocess())
            
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            model = MuSc(config, seed=42)
            
            final_memory = torch.cuda.memory_allocated()
            
            # 内存使用应该是合理的
            memory_increase = final_memory - initial_memory
            assert memory_increase < 5e8  # 不应该超过500MB