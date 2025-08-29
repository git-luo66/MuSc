"""
测试MSM (Mutual Scoring Module) 模块
"""
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from models.modules._MSM import compute_scores_fast, compute_scores_slow, MSM


class TestComputeScoresFast:
    """测试快速分数计算函数"""
    
    def test_compute_scores_fast_basic(self, device):
        """测试基本功能"""
        # 创建测试数据
        Z = torch.randn(5, 10, 64).to(device)  # 5个图像，10个patch，64维特征
        i = 2  # 计算第2个图像的分数
        
        scores = compute_scores_fast(Z, i, device)
        
        assert scores.shape == (10,)  # 返回10个patch的分数
        assert scores.dtype == torch.float32
        assert torch.all(scores >= 0)  # 距离应该非负
    
    def test_compute_scores_fast_different_params(self, device):
        """测试不同参数设置"""
        Z = torch.randn(8, 15, 128).to(device)
        i = 3
        
        # 测试不同的topmin参数
        scores1 = compute_scores_fast(Z, i, device, topmin_min=0, topmin_max=0.2)
        scores2 = compute_scores_fast(Z, i, device, topmin_min=0.1, topmin_max=0.4)
        
        assert scores1.shape == scores2.shape == (15,)
        # 不同参数应该产生不同的结果
        assert not torch.allclose(scores1, scores2)
    
    def test_compute_scores_fast_edge_cases(self, device):
        """测试边界情况"""
        # 测试最小图像数量
        Z = torch.randn(2, 5, 32).to(device)
        scores = compute_scores_fast(Z, 0, device)
        assert scores.shape == (5,)
        
        # 测试单个patch
        Z = torch.randn(3, 1, 64).to(device)
        scores = compute_scores_fast(Z, 1, device)
        assert scores.shape == (1,)
    
    def test_compute_scores_fast_topmin_values(self, device):
        """测试topmin参数的不同值"""
        Z = torch.randn(6, 20, 64).to(device)
        i = 2
        
        # 测试百分比值
        scores1 = compute_scores_fast(Z, i, device, topmin_min=0.1, topmin_max=0.3)
        
        # 测试绝对值
        scores2 = compute_scores_fast(Z, i, device, topmin_min=1, topmin_max=3)
        
        assert scores1.shape == scores2.shape == (20,)


class TestComputeScoresSlow:
    """测试慢速分数计算函数"""
    
    def test_compute_scores_slow_basic(self, device):
        """测试基本功能"""
        Z = torch.randn(4, 8, 32).to(device)
        i = 1
        
        scores = compute_scores_slow(Z, i, device)
        
        assert scores.shape == (8,)
        assert scores.dtype == torch.float32
        assert torch.all(scores >= 0)
    
    def test_compute_scores_consistency(self, device):
        """测试快速和慢速算法的一致性"""
        # 使用小数据集确保结果一致
        Z = torch.randn(3, 5, 16).to(device)
        i = 1
        
        scores_fast = compute_scores_fast(Z, i, device)
        scores_slow = compute_scores_slow(Z, i, device)
        
        # 两种算法应该产生相同或非常接近的结果
        assert torch.allclose(scores_fast, scores_slow, rtol=1e-4)


class TestMSM:
    """测试MSM主函数"""
    
    def test_msm_basic(self, device):
        """测试基本MSM计算"""
        Z = torch.randn(4, 10, 64).to(device)
        
        with patch('models.modules._MSM.tqdm', side_effect=lambda x: x):  # 禁用tqdm进度条
            result = MSM(Z, device)
        
        assert result.shape == (4, 10)  # 4个图像，每个10个patch分数
        assert result.dtype == torch.float64  # 应该是double类型
        assert torch.all(result >= 0)
    
    def test_msm_different_sizes(self, device):
        """测试不同输入尺寸"""
        # 测试不同的图像数量和patch数量
        test_cases = [
            (3, 5, 32),
            (6, 15, 128),
            (2, 8, 256)
        ]
        
        for img_num, patch_num, feat_dim in test_cases:
            Z = torch.randn(img_num, patch_num, feat_dim).to(device)
            
            with patch('models.modules._MSM.tqdm', side_effect=lambda x: x):
                result = MSM(Z, device)
            
            assert result.shape == (img_num, patch_num)
    
    def test_msm_parameters(self, device):
        """测试MSM参数"""
        Z = torch.randn(3, 8, 64).to(device)
        
        with patch('models.modules._MSM.tqdm', side_effect=lambda x: x):
            result1 = MSM(Z, device, topmin_min=0, topmin_max=0.2)
            result2 = MSM(Z, device, topmin_min=0.1, topmin_max=0.4)
        
        assert result1.shape == result2.shape == (3, 8)
        # 不同参数应该产生不同结果
        assert not torch.allclose(result1, result2)
    
    @pytest.mark.slow
    def test_msm_large_data(self, device):
        """测试大数据集（标记为慢测试）"""
        Z = torch.randn(20, 50, 512).to(device)
        
        with patch('models.modules._MSM.tqdm', side_effect=lambda x: x):
            result = MSM(Z, device)
        
        assert result.shape == (20, 50)
        assert torch.all(torch.isfinite(result))  # 确保没有NaN或Inf
    
    def test_msm_memory_efficiency(self, device):
        """测试内存使用"""
        if not torch.cuda.is_available():
            pytest.skip("需要CUDA来测试内存使用")
        
        Z = torch.randn(10, 20, 256).to(device)
        
        # 记录初始内存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        with patch('models.modules._MSM.tqdm', side_effect=lambda x: x):
            result = MSM(Z, device)
        
        final_memory = torch.cuda.memory_allocated()
        
        # 验证结果正确性
        assert result.shape == (10, 20)
        
        # 内存使用应该是合理的（不应该无限增长）
        memory_increase = final_memory - initial_memory
        assert memory_increase < 1e9  # 不应该超过1GB
    
    def test_msm_deterministic(self, device):
        """测试结果的确定性"""
        torch.manual_seed(42)
        Z1 = torch.randn(3, 5, 32).to(device)
        
        torch.manual_seed(42)
        Z2 = torch.randn(3, 5, 32).to(device)
        
        with patch('models.modules._MSM.tqdm', side_effect=lambda x: x):
            result1 = MSM(Z1, device)
            result2 = MSM(Z2, device)
        
        # 相同的输入应该产生相同的结果
        assert torch.allclose(result1, result2)