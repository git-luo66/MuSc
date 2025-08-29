"""
测试RsCIN (Relation-guided Scoring in Class-agnostic Image-level Novelty) 模块
"""
import pytest
import torch
import numpy as np
from models.modules._RsCIN import MMO, RsCIN


class TestMMO:
    """测试MMO (Matrix Masking Operation) 函数"""
    
    def test_mmo_basic(self):
        """测试基本MMO计算"""
        # 创建测试数据
        W = torch.tensor([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])
        score = torch.tensor([0.5, 0.7, 0.3])
        k_list = [1]
        
        result = MMO(W, score, k_list)
        
        assert result.shape == (3,)
        assert torch.all(torch.isfinite(result))
        assert result.dtype == torch.float32
    
    def test_mmo_multiple_k_values(self):
        """测试多个k值的MMO计算"""
        W = torch.rand(5, 5)
        # 确保对角线为1（相似度矩阵的性质）
        W = W + W.T
        W.fill_diagonal_(1.0)
        
        score = torch.rand(5)
        k_list = [1, 2, 3]
        
        result = MMO(W, score, k_list)
        
        assert result.shape == (5,)
        assert torch.all(torch.isfinite(result))
    
    def test_mmo_edge_cases(self):
        """测试边界情况"""
        # 测试最小矩阵
        W = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        score = torch.tensor([0.3, 0.7])
        k_list = [1]
        
        result = MMO(W, score, k_list)
        assert result.shape == (2,)
        
        # 测试k值等于矩阵大小-1
        k_list = [1]  # 对于2x2矩阵，最大k值为1
        result = MMO(W, score, k_list)
        assert result.shape == (2,)
    
    def test_mmo_identity_matrix(self):
        """测试单位矩阵的情况"""
        W = torch.eye(4)
        score = torch.tensor([0.2, 0.4, 0.6, 0.8])
        k_list = [1, 2]
        
        result = MMO(W, score, k_list)
        
        assert result.shape == (4,)
        assert torch.all(torch.isfinite(result))
    
    def test_mmo_zero_scores(self):
        """测试零分数的情况"""
        W = torch.rand(3, 3)
        W = W + W.T
        W.fill_diagonal_(1.0)
        
        score = torch.zeros(3)
        k_list = [1]
        
        result = MMO(W, score, k_list)
        
        assert result.shape == (3,)
        assert torch.allclose(result, torch.zeros(3))


class TestRsCIN:
    """测试RsCIN主函数"""
    
    def test_rscin_with_cls_tokens(self):
        """测试使用类别token的RsCIN"""
        scores_old = np.array([0.2, 0.5, 0.8, 0.3, 0.7])
        cls_tokens = np.random.randn(5, 64)  # 5个图像，64维特征
        k_list = [1, 2]
        
        result = RsCIN(scores_old, cls_tokens, k_list)
        
        assert result.shape == (5,)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # 归一化后应该非负
        assert np.all(result <= 1)  # 归一化后应该不超过1
    
    def test_rscin_without_cls_tokens(self):
        """测试没有类别token的情况"""
        scores_old = np.array([0.2, 0.5, 0.8, 0.3])
        
        # cls_tokens为None时应该返回原始分数
        result = RsCIN(scores_old, None, [1, 2])
        np.testing.assert_array_equal(result, scores_old)
        
        # k_list包含0时也应该返回原始分数
        cls_tokens = np.random.randn(4, 32)
        result = RsCIN(scores_old, cls_tokens, [0, 1])
        np.testing.assert_array_equal(result, scores_old)
    
    def test_rscin_k_list_with_zero(self):
        """测试k_list包含0的情况"""
        scores_old = np.array([0.1, 0.4, 0.9])
        cls_tokens = np.random.randn(3, 16)
        k_list = [0, 1, 2]
        
        result = RsCIN(scores_old, cls_tokens, k_list)
        np.testing.assert_array_equal(result, scores_old)
    
    def test_rscin_normalization(self):
        """测试分数归一化"""
        # 创建具有已知范围的分数
        scores_old = np.array([10.0, 20.0, 30.0, 15.0])
        cls_tokens = np.random.randn(4, 32)
        k_list = [1]
        
        result = RsCIN(scores_old, cls_tokens, k_list)
        
        # 结果应该在[0,1]范围内
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert result.shape == scores_old.shape
    
    def test_rscin_identical_tokens(self):
        """测试相同token的情况"""
        scores_old = np.array([0.2, 0.5, 0.8])
        # 创建相同的token
        cls_tokens = np.ones((3, 16))
        k_list = [1]
        
        result = RsCIN(scores_old, cls_tokens, k_list)
        
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
    
    def test_rscin_single_image(self):
        """测试单个图像的情况"""
        scores_old = np.array([0.5])
        cls_tokens = np.random.randn(1, 32)
        k_list = [1]
        
        # 单个图像时，k_list中的k值会被限制
        # 这可能导致特殊行为，需要确保不出错
        try:
            result = RsCIN(scores_old, cls_tokens, k_list)
            assert result.shape == (1,)
        except Exception as e:
            # 如果单个图像无法处理，这也是可以接受的
            assert "index" in str(e).lower() or "dimension" in str(e).lower()
    
    def test_rscin_large_k_values(self):
        """测试大k值的情况"""
        scores_old = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        cls_tokens = np.random.randn(5, 64)
        k_list = [3, 4]  # 接近图像数量的k值
        
        result = RsCIN(scores_old, cls_tokens, k_list)
        
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))
    
    def test_rscin_data_types(self):
        """测试不同数据类型"""
        # 测试float32输入
        scores_old = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        cls_tokens = np.random.randn(3, 32).astype(np.float32)
        k_list = [1]
        
        result = RsCIN(scores_old, cls_tokens, k_list)
        assert isinstance(result, np.ndarray)
        
        # 测试float64输入
        scores_old = np.array([0.2, 0.5, 0.8], dtype=np.float64)
        cls_tokens = np.random.randn(3, 32).astype(np.float64)
        
        result = RsCIN(scores_old, cls_tokens, k_list)
        assert isinstance(result, np.ndarray)
    
    def test_rscin_reproducibility(self):
        """测试结果的可重现性"""
        np.random.seed(42)
        scores_old1 = np.array([0.2, 0.5, 0.8, 0.3])
        cls_tokens1 = np.random.randn(4, 32)
        
        np.random.seed(42)
        scores_old2 = np.array([0.2, 0.5, 0.8, 0.3])
        cls_tokens2 = np.random.randn(4, 32)
        
        k_list = [1, 2]
        
        result1 = RsCIN(scores_old1, cls_tokens1, k_list)
        result2 = RsCIN(scores_old2, cls_tokens2, k_list)
        
        np.testing.assert_array_almost_equal(result1, result2)