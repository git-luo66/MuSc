"""
测试评估指标模块
"""
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.metrics import cal_pro_score, compute_metrics


class TestCalProScore:
    """测试PRO分数计算函数"""
    
    def test_cal_pro_score_basic(self):
        """测试基本PRO分数计算"""
        # 创建简单的测试数据
        masks = np.array([
            [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 1, 1], [0, 1, 1]]
        ])
        amaps = np.array([
            [[0.8, 0.9, 0.1], [0.7, 0.8, 0.2], [0.1, 0.1, 0.1]],
            [[0.2, 0.1, 0.9], [0.1, 0.8, 0.9], [0.1, 0.7, 0.8]]
        ])
        
        score = cal_pro_score(masks, amaps, max_step=10)
        assert isinstance(score, (float, np.floating))
        assert 0 <= score <= 1
    
    def test_cal_pro_score_empty_masks(self):
        """测试空掩码的情况"""
        masks = np.zeros((2, 3, 3))
        amaps = np.random.rand(2, 3, 3)
        
        score = cal_pro_score(masks, amaps)
        assert isinstance(score, (float, np.floating))
    
    def test_cal_pro_score_all_anomaly_masks(self):
        """测试全异常掩码的情况"""
        masks = np.ones((2, 3, 3))
        amaps = np.random.rand(2, 3, 3)
        
        score = cal_pro_score(masks, amaps)
        assert isinstance(score, (float, np.floating))
        assert 0 <= score <= 1
    
    def test_cal_pro_score_parameters(self):
        """测试不同参数设置"""
        masks = np.random.randint(0, 2, (3, 5, 5))
        amaps = np.random.rand(3, 5, 5)
        
        # 测试不同的max_step
        score1 = cal_pro_score(masks, amaps, max_step=50)
        score2 = cal_pro_score(masks, amaps, max_step=100)
        assert isinstance(score1, (float, np.floating))
        assert isinstance(score2, (float, np.floating))
        
        # 测试不同的expect_fpr
        score3 = cal_pro_score(masks, amaps, expect_fpr=0.1)
        score4 = cal_pro_score(masks, amaps, expect_fpr=0.5)
        assert isinstance(score3, (float, np.floating))
        assert isinstance(score4, (float, np.floating))


class TestComputeMetrics:
    """测试指标计算函数"""
    
    def test_compute_metrics_normal_case(self):
        """测试正常情况的指标计算"""
        # 创建测试数据
        gt_sp = np.array([0, 0, 1, 1, 0, 1])  # 图像级真值
        pr_sp = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.7])  # 图像级预测
        gt_px = np.random.randint(0, 2, (2, 10, 10))  # 像素级真值
        pr_px = np.random.rand(2, 10, 10)  # 像素级预测
        
        image_metrics, pixel_metrics = compute_metrics(gt_sp, pr_sp, gt_px, pr_px)
        
        # 验证返回值格式
        assert len(image_metrics) == 3  # [auroc_sp, f1_sp, ap_sp]
        assert len(pixel_metrics) == 4  # [auroc_px, f1_px, ap_px, aupro]
        
        # 验证指标值范围
        for metric in image_metrics:
            assert 0 <= metric <= 1
        for metric in pixel_metrics:
            assert 0 <= metric <= 1
    
    def test_compute_metrics_no_anomalies(self):
        """测试没有异常的情况"""
        gt_sp = np.array([0, 0, 0, 0])  # 全部正常
        pr_sp = np.array([0.1, 0.2, 0.3, 0.4])
        
        image_metrics, pixel_metrics = compute_metrics(gt_sp=gt_sp, pr_sp=pr_sp)
        
        # 当没有异常时，应该返回0值
        assert image_metrics == [0, 0, 0]
        assert pixel_metrics == [0, 0, 0, 0]
    
    def test_compute_metrics_all_anomalies(self):
        """测试全部异常的情况"""
        gt_sp = np.array([1, 1, 1, 1])  # 全部异常
        pr_sp = np.array([0.6, 0.7, 0.8, 0.9])
        
        image_metrics, pixel_metrics = compute_metrics(gt_sp=gt_sp, pr_sp=pr_sp)
        
        # 当全部都是异常时，应该返回0值
        assert image_metrics == [0, 0, 0]
        assert pixel_metrics == [0, 0, 0, 0]
    
    def test_compute_metrics_none_inputs(self):
        """测试None输入的情况"""
        image_metrics, pixel_metrics = compute_metrics()
        
        assert image_metrics == [0, 0, 0]
        assert pixel_metrics == [0, 0, 0, 0]
    
    def test_compute_metrics_pixel_only(self):
        """测试只有像素级数据的情况"""
        gt_px = np.random.randint(0, 2, (3, 5, 5))
        pr_px = np.random.rand(3, 5, 5)
        
        image_metrics, pixel_metrics = compute_metrics(gt_px=gt_px, pr_px=pr_px)
        
        assert image_metrics == [0, 0, 0]
        assert len(pixel_metrics) == 4
        for metric in pixel_metrics:
            assert 0 <= metric <= 1
    
    def test_compute_metrics_image_only(self):
        """测试只有图像级数据的情况"""
        gt_sp = np.array([0, 1, 0, 1, 1])
        pr_sp = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        
        image_metrics, pixel_metrics = compute_metrics(gt_sp=gt_sp, pr_sp=pr_sp)
        
        assert len(image_metrics) == 3
        assert pixel_metrics == [0, 0, 0, 0]
        for metric in image_metrics:
            assert 0 <= metric <= 1
    
    def test_compute_metrics_edge_cases(self):
        """测试边界情况"""
        # 测试单个样本
        gt_sp = np.array([1])
        pr_sp = np.array([0.8])
        
        image_metrics, pixel_metrics = compute_metrics(gt_sp=gt_sp, pr_sp=pr_sp)
        assert image_metrics == [0, 0, 0]  # 单个样本无法计算AUC
        
        # 测试空数组
        gt_sp = np.array([])
        pr_sp = np.array([])
        
        image_metrics, pixel_metrics = compute_metrics(gt_sp=gt_sp, pr_sp=pr_sp)
        assert image_metrics == [0, 0, 0]
        assert pixel_metrics == [0, 0, 0, 0]
    
    @patch('utils.metrics.cal_pro_score')
    def test_compute_metrics_with_mocked_pro_score(self, mock_pro_score):
        """测试使用模拟PRO分数计算"""
        mock_pro_score.return_value = 0.85
        
        gt_sp = np.array([0, 1, 0, 1])
        pr_sp = np.array([0.2, 0.8, 0.3, 0.9])
        gt_px = np.random.randint(0, 2, (2, 5, 5))
        pr_px = np.random.rand(2, 5, 5)
        
        image_metrics, pixel_metrics = compute_metrics(gt_sp, pr_sp, gt_px, pr_px)
        
        # 验证PRO分数被正确调用和返回
        mock_pro_score.assert_called_once()
        assert pixel_metrics[3] == 0.85  # aupro值
    
    def test_compute_metrics_data_types(self):
        """测试不同数据类型的输入"""
        # 测试torch tensor输入
        gt_sp_torch = torch.tensor([0, 1, 0, 1])
        pr_sp_torch = torch.tensor([0.2, 0.8, 0.3, 0.9])
        
        # 应该能够处理torch tensor（转换为numpy）
        try:
            image_metrics, pixel_metrics = compute_metrics(
                gt_sp_torch.numpy(), pr_sp_torch.numpy()
            )
            assert len(image_metrics) == 3
            assert len(pixel_metrics) == 4
        except Exception as e:
            pytest.fail(f"应该能够处理torch tensor输入: {e}")
    
    def test_compute_metrics_perfect_prediction(self):
        """测试完美预测的情况"""
        gt_sp = np.array([0, 0, 1, 1])
        pr_sp = np.array([0.0, 0.0, 1.0, 1.0])  # 完美预测
        
        image_metrics, pixel_metrics = compute_metrics(gt_sp=gt_sp, pr_sp=pr_sp)
        
        # AUC应该为1.0，AP也应该为1.0
        assert image_metrics[0] == 1.0  # auroc_sp
        assert image_metrics[2] == 1.0  # ap_sp