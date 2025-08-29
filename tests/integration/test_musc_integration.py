"""
MuSc模型集成测试
测试模型的完整工作流程
"""
import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, Mock
from models.musc import MuSc


class TestMuScIntegration:
    """MuSc模型集成测试"""
    
    @pytest.fixture
    def integration_config(self, temp_dir):
        """集成测试配置"""
        return {
            'datasets': {
                'dataset_name': 'mvtec_ad',
                'data_path': temp_dir,
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
            'device': '0' if torch.cuda.is_available() else 'cpu',
            'testing': {
                'output_dir': os.path.join(temp_dir, 'output'),
                'vis': False,
                'vis_type': 'single_norm',
                'save_excel': False
            }
        }
    
    @pytest.fixture
    def mock_dataset(self):
        """模拟数据集"""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=4)
        
        # 模拟数据集返回值
        def mock_getitem(idx):
            return (
                torch.randn(3, 224, 224),  # image
                torch.randint(0, 2, (224, 224)),  # mask  
                idx % 2,  # label (0=normal, 1=anomaly)
                f"test_image_{idx}.png"  # image_path
            )
        dataset.__getitem__ = mock_getitem
        return dataset
    
    def test_musc_initialization(self, integration_config):
        """测试MuSc模型初始化"""
        model = MuSc(integration_config, seed=42)
        
        assert model.cfg == integration_config
        assert model.seed == 42
        assert model.dataset == 'mvtec_ad'
        assert model.categories == 'bottle'
        assert isinstance(model.device, torch.device)
    
    @patch('models.musc.mvtec.MVTecDataset')
    @patch('models.musc._backbones.load')
    def test_musc_load_datasets(self, mock_backbone_load, mock_dataset_class, integration_config, mock_dataset):
        """测试数据集加载"""
        mock_dataset_class.return_value = mock_dataset
        mock_backbone_load.return_value = (MagicMock(), MagicMock())
        
        model = MuSc(integration_config, seed=42)
        dataset = model.load_datasets('bottle')
        
        assert dataset is not None
        mock_dataset_class.assert_called_once()
    
    @patch('models.musc.visa.VisaDataset')
    @patch('models.musc._backbones.load')
    def test_musc_load_visa_dataset(self, mock_backbone_load, mock_dataset_class, integration_config, mock_dataset):
        """测试VisA数据集加载"""
        mock_dataset_class.return_value = mock_dataset
        mock_backbone_load.return_value = (MagicMock(), MagicMock())
        
        # 修改配置为visa数据集
        integration_config['datasets']['dataset_name'] = 'visa'
        model = MuSc(integration_config, seed=42)
        dataset = model.load_datasets('candle')
        
        assert dataset is not None
        mock_dataset_class.assert_called_once()
    
    @patch('models.musc.btad.BTADDataset')  
    @patch('models.musc._backbones.load')
    def test_musc_load_btad_dataset(self, mock_backbone_load, mock_dataset_class, integration_config, mock_dataset):
        """测试BTAD数据集加载"""
        mock_dataset_class.return_value = mock_dataset
        mock_backbone_load.return_value = (MagicMock(), MagicMock())
        
        # 修改配置为btad数据集
        integration_config['datasets']['dataset_name'] = 'btad'
        model = MuSc(integration_config, seed=42)
        dataset = model.load_datasets('01')
        
        assert dataset is not None
        mock_dataset_class.assert_called_once()
    
    def test_musc_visualization_single_norm(self, integration_config, temp_dir):
        """测试单图归一化可视化"""
        integration_config['testing']['vis'] = True
        integration_config['testing']['vis_type'] = 'single_norm'
        
        model = MuSc(integration_config, seed=42)
        
        # 创建测试数据
        image_path_list = [
            os.path.join(temp_dir, 'defect_type', 'test_image.png')
        ]
        gt_list = [1]  # 异常图像
        pr_px = np.random.rand(1, 224, 224)
        
        # 创建目录结构
        os.makedirs(os.path.dirname(image_path_list[0]), exist_ok=True)
        
        with patch('cv2.imwrite') as mock_imwrite:
            model.visualization(image_path_list, gt_list, pr_px, 'bottle')
            mock_imwrite.assert_called_once()
    
    def test_musc_visualization_whole_norm(self, integration_config, temp_dir):
        """测试全局归一化可视化"""
        integration_config['testing']['vis'] = True
        integration_config['testing']['vis_type'] = 'whole_norm'
        
        model = MuSc(integration_config, seed=42)
        
        # 创建测试数据
        image_path_list = [
            os.path.join(temp_dir, 'good', 'normal_image.png'),
            os.path.join(temp_dir, 'defect', 'abnormal_image.png')
        ]
        gt_list = [0, 1]
        pr_px = np.random.rand(2, 224, 224)
        
        # 创建目录结构
        for path in image_path_list:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with patch('cv2.imwrite') as mock_imwrite:
            model.visualization(image_path_list, gt_list, pr_px, 'bottle')
            assert mock_imwrite.call_count == 2
    
    @patch('models.musc._backbones.load')
    def test_musc_categories_handling(self, mock_backbone_load, integration_config):
        """测试类别处理"""
        mock_backbone_load.return_value = (MagicMock(), MagicMock())
        
        # 测试单个类别
        integration_config['datasets']['class_name'] = 'bottle'
        model = MuSc(integration_config)
        assert model.categories == 'bottle'
        
        # 测试ALL类别
        integration_config['datasets']['class_name'] = 'ALL'
        model = MuSc(integration_config)
        assert isinstance(model.categories, list)
        assert len(model.categories) > 0
    
    @patch('models.musc.torch.utils.data.DataLoader')
    @patch('models.musc.MVTecDataset')
    @patch('models.musc._backbones.load')
    def test_musc_make_category_data_flow(self, mock_backbone_load, mock_dataset_class, 
                                         mock_dataloader, integration_config, mock_dataset):
        """测试make_category_data的基本流程"""
        # 设置模拟对象
        mock_backbone_load.return_value = (MagicMock(), MagicMock())
        mock_dataset_class.return_value = mock_dataset
        
        # 模拟dataloader
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__iter__ = MagicMock(return_value=iter([
            (torch.randn(2, 3, 224, 224), torch.randint(0, 2, (2, 224, 224)), 
             torch.tensor([0, 1]), ['img1.png', 'img2.png'])
        ]))
        mock_dataloader.return_value = mock_dataloader_instance
        
        model = MuSc(integration_config, seed=42)
        
        # 模拟骨干网络输出
        with patch.object(model, 'backbone') as mock_backbone:
            mock_backbone.return_value = [torch.randn(2, 197, 768), torch.randn(2, 197, 768)]
            
            # 模拟LNAMD输出
            with patch('models.musc.LNAMD') as mock_lnamd_class:
                mock_lnamd = MagicMock()
                mock_lnamd._embed.return_value = torch.randn(2, 196, 2, 768)
                mock_lnamd_class.return_value = mock_lnamd
                
                # 模拟MSM输出
                with patch('models.musc.MSM') as mock_msm:
                    mock_msm.return_value = torch.randn(2, 196)
                    
                    # 模拟RsCIN输出
                    with patch('models.musc.RsCIN') as mock_rscin:
                        mock_rscin.return_value = np.random.rand(2)
                        
                        try:
                            result = model.make_category_data('bottle')
                            # 验证函数能够执行完成
                            assert True  # 如果到达这里说明没有异常
                        except Exception as e:
                            # 记录具体错误信息
                            pytest.fail(f"make_category_data执行失败: {e}")
    
    def test_musc_device_handling(self, integration_config):
        """测试设备处理"""
        # 测试CUDA设备
        if torch.cuda.is_available():
            integration_config['device'] = '0'
            model = MuSc(integration_config)
            assert model.device.type == 'cuda'
        
        # 测试CPU设备
        integration_config['device'] = 'cpu'
        model = MuSc(integration_config)
        assert model.device.type == 'cpu'
    
    def test_musc_output_directory_creation(self, integration_config, temp_dir):
        """测试输出目录创建"""
        output_dir = os.path.join(temp_dir, 'test_output')
        integration_config['testing']['output_dir'] = output_dir
        
        model = MuSc(integration_config)
        
        # 输出目录应该被创建
        assert os.path.exists(output_dir)
    
    @pytest.mark.slow
    @patch('models.musc._backbones.load')
    def test_musc_full_pipeline_mock(self, mock_backbone_load, integration_config):
        """测试完整流水线（使用模拟）"""
        # 模拟骨干网络
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_backbone_load.return_value = (mock_model, mock_preprocess)
        
        model = MuSc(integration_config, seed=42)
        
        # 验证模型属性设置正确
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'preprocess')
        assert model.batch_size == 2
        assert model.image_size == 224