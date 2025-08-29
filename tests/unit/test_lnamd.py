"""
测试LNAMD (Locally Neighborhood Aware Anomaly Detection) 模块
"""
import pytest
import torch
import numpy as np
from models.modules._LNAMD import PatchMaker, Preprocessing, MeanMapper, LNAMD


class TestPatchMaker:
    """测试PatchMaker类"""
    
    def test_patchmaker_init(self):
        """测试PatchMaker初始化"""
        patchmaker = PatchMaker(patchsize=3, stride=1)
        assert patchmaker.patchsize == 3
        assert patchmaker.stride == 1
        
        patchmaker_no_stride = PatchMaker(patchsize=5)
        assert patchmaker_no_stride.patchsize == 5
        assert patchmaker_no_stride.stride is None
    
    def test_patchify_basic(self):
        """测试基本patch化功能"""
        patchmaker = PatchMaker(patchsize=3, stride=1)
        features = torch.randn(2, 64, 8, 8)  # batch=2, channels=64, H=8, W=8
        
        patches = patchmaker.patchify(features)
        
        # 验证输出形状
        expected_patches = 8 * 8  # 对于8x8输入，stride=1，patchsize=3
        assert patches.shape[0] == 2  # batch size
        assert patches.shape[1] == expected_patches
        assert patches.shape[2] == 64  # channels
        assert patches.shape[3] == 3  # patchsize
        assert patches.shape[4] == 3  # patchsize
    
    def test_patchify_with_spatial_info(self):
        """测试返回空间信息的patch化"""
        patchmaker = PatchMaker(patchsize=3, stride=2)
        features = torch.randn(1, 32, 6, 6)
        
        patches, spatial_info = patchmaker.patchify(features, return_spatial_info=True)
        
        assert isinstance(spatial_info, list)
        assert len(spatial_info) == 2  # H和W的patch数量
        assert all(isinstance(x, int) for x in spatial_info)
        assert patches.shape[1] == spatial_info[0] * spatial_info[1]
    
    def test_patchify_different_strides(self):
        """测试不同stride值"""
        features = torch.randn(1, 16, 10, 10)
        
        patchmaker1 = PatchMaker(patchsize=3, stride=1)
        patchmaker2 = PatchMaker(patchsize=3, stride=2)
        
        patches1 = patchmaker1.patchify(features)
        patches2 = patchmaker2.patchify(features)
        
        # stride=2应该产生更少的patches
        assert patches2.shape[1] < patches1.shape[1]
    
    def test_patchify_different_patchsizes(self):
        """测试不同patch大小"""
        features = torch.randn(1, 16, 8, 8)
        
        patchmaker_small = PatchMaker(patchsize=3, stride=1)
        patchmaker_large = PatchMaker(patchsize=5, stride=1)
        
        patches_small = patchmaker_small.patchify(features)
        patches_large = patchmaker_large.patchify(features)
        
        # 验证patch尺寸
        assert patches_small.shape[3] == 3
        assert patches_large.shape[3] == 5


class TestMeanMapper:
    """测试MeanMapper类"""
    
    def test_meanmapper_init(self):
        """测试MeanMapper初始化"""
        mapper = MeanMapper(output_dim=256)
        assert hasattr(mapper, 'preprocessing')
        assert isinstance(mapper.preprocessing, torch.nn.AdaptiveAvgPool2d)
    
    def test_meanmapper_forward(self):
        """测试MeanMapper前向传播"""
        mapper = MeanMapper(output_dim=128)
        input_tensor = torch.randn(4, 256, 7, 7)  # batch=4, channels=256, H=7, W=7
        
        output = mapper(input_tensor)
        
        assert output.shape == (4, 256, 1, 1)  # 应该被池化到1x1
        assert output.dtype == input_tensor.dtype
    
    def test_meanmapper_different_input_sizes(self):
        """测试不同输入尺寸"""
        mapper = MeanMapper(output_dim=64)
        
        # 测试不同的空间尺寸
        inputs = [
            torch.randn(2, 128, 4, 4),
            torch.randn(2, 128, 8, 8),
            torch.randn(2, 128, 16, 16)
        ]
        
        for input_tensor in inputs:
            output = mapper(input_tensor)
            assert output.shape == (2, 128, 1, 1)


class TestPreprocessing:
    """测试Preprocessing类"""
    
    def test_preprocessing_init(self):
        """测试Preprocessing初始化"""
        input_layers = [256, 512, 1024]
        output_dim = 768
        
        preprocessing = Preprocessing(input_layers, output_dim)
        
        assert len(preprocessing.preprocessing_modules) == 3
        assert all(isinstance(module, MeanMapper) for module in preprocessing.preprocessing_modules)
    
    def test_preprocessing_forward(self):
        """测试Preprocessing前向传播"""
        input_layers = [128, 256]
        output_dim = 512
        
        preprocessing = Preprocessing(input_layers, output_dim)
        
        # 创建对应的特征列表
        features = [
            torch.randn(2, 128, 7, 7),
            torch.randn(2, 256, 7, 7)
        ]
        
        output = preprocessing(features)
        
        assert len(output) == 2
        assert all(feat.shape == (2, feat.shape[1], 1, 1) for feat in output)
    
    def test_preprocessing_single_layer(self):
        """测试单层预处理"""
        preprocessing = Preprocessing([512], 256)
        features = [torch.randn(3, 512, 5, 5)]
        
        output = preprocessing(features)
        
        assert len(output) == 1
        assert output[0].shape == (3, 512, 1, 1)


class TestLNAMDModule:
    """测试LNAMD模块类"""
    
    def test_lnamd_module_init(self, device):
        """测试LNAMD模块初始化"""
        lnamd = LNAMD(device=device, feature_dim=512, feature_layer=[1, 2], r=3)
        
        assert lnamd.device == device
        assert lnamd.r == 3
        assert isinstance(lnamd.patch_maker, PatchMaker)
        assert isinstance(lnamd.LNA, Preprocessing)
    
    def test_lnamd_embed_basic(self, device):
        """测试LNAMD嵌入功能"""
        lnamd = LNAMD(device=device, feature_dim=256, feature_layer=[1, 2], r=1)
        
        # 创建包含CLS token的特征（模拟ViT输出）
        features = [
            torch.randn(2, 197, 256).to(device),  # 196 patches + 1 CLS token
            torch.randn(2, 197, 256).to(device)
        ]
        
        result = lnamd._embed(features)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 2  # batch size
        assert result.device.type == "cpu"  # 应该返回到CPU
    
    def test_lnamd_embed_different_r_values(self, device):
        """测试不同r值的嵌入"""
        features = [torch.randn(1, 197, 128).to(device)]
        
        lnamd_r1 = LNAMD(device=device, feature_dim=128, feature_layer=[1], r=1)
        lnamd_r3 = LNAMD(device=device, feature_dim=128, feature_layer=[1], r=3)
        
        result_r1 = lnamd_r1._embed(features)
        result_r3 = lnamd_r3._embed(features)
        
        # 不同r值应该产生不同的结果
        assert result_r1.shape[0] == result_r3.shape[0] == 1
        # r=3应该产生更少的patches（因为patch size更大）
        if result_r1.shape[1] != result_r3.shape[1]:
            assert result_r3.shape[1] < result_r1.shape[1]
    
    def test_lnamd_embed_multiple_layers(self, device):
        """测试多层特征嵌入"""
        lnamd = LNAMD(device=device, feature_dim=256, feature_layer=[1, 2, 3], r=2)
        
        features = [
            torch.randn(1, 197, 256).to(device),
            torch.randn(1, 197, 256).to(device),
            torch.randn(1, 197, 256).to(device)
        ]
        
        result = lnamd._embed(features)
        
        assert result.shape[0] == 1
        assert result.shape[-1] == 3  # 3个特征层
    
    def test_lnamd_embed_cls_token_removal(self, device):
        """测试CLS token移除"""
        lnamd = LNAMD(device=device, feature_dim=64, feature_layer=[1], r=1)
        
        # 创建包含CLS token的特征
        features_with_cls = [torch.randn(1, 50, 64).to(device)]  # 49 patches + 1 CLS
        
        result = lnamd._embed(features_with_cls)
        
        # 验证CLS token被正确移除（应该有49个patches，形成7x7的网格）
        expected_patches = 49  # 7x7
        assert result.shape[1] == expected_patches
    
    @pytest.mark.gpu
    def test_lnamd_gpu_processing(self):
        """测试GPU处理"""
        if not torch.cuda.is_available():
            pytest.skip("需要CUDA进行GPU测试")
        
        device = torch.device("cuda:0")
        lnamd = LNAMD(device=device, feature_dim=128, feature_layer=[1], r=2)
        
        features = [torch.randn(1, 197, 128).to(device)]
        
        result = lnamd._embed(features)
        
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"  # 最终结果应该在CPU上
    
    def test_lnamd_different_r_values(self, sample_features_list, device):
        """测试不同r值"""
        r_list_small = [1]
        r_list_large = [1, 3, 5]
        
        result_small = LNAMD(sample_features_list, r_list_small, device)
        result_large = LNAMD(sample_features_list, r_list_large, device)
        
        # 不同r值应该产生相同形状但不同值的结果
        assert result_small.shape == result_large.shape
        assert not torch.allclose(result_small, result_large)
    
    def test_lnamd_single_feature_layer(self, device):
        """测试单个特征层"""
        features_list = [torch.randn(2, 128, 8, 8)]
        r_list = [1, 2]
        
        result = LNAMD(features_list, r_list, device)
        
        assert result.shape[0] == 2
        assert result.device.type == device.type
    
    def test_lnamd_empty_r_list(self, sample_features_list, device):
        """测试空r_list的情况"""
        r_list = []
        
        # 空r_list可能会导致错误或返回特殊值
        try:
            result = LNAMD(sample_features_list, r_list, device)
            # 如果成功，验证基本属性
            assert isinstance(result, torch.Tensor)
            assert result.device.type == device.type
        except (ValueError, IndexError):
            # 空r_list导致错误是可以接受的
            pass
    
    @pytest.mark.gpu
    def test_lnamd_gpu_computation(self, sample_features_list):
        """测试GPU计算（需要GPU标记）"""
        if not torch.cuda.is_available():
            pytest.skip("需要CUDA进行GPU测试")
        
        device = torch.device("cuda:0")
        # 将特征移动到GPU
        gpu_features = [feat.to(device) for feat in sample_features_list]
        r_list = [1, 3]
        
        result = LNAMD(gpu_features, r_list, device)
        
        assert result.device.type == "cuda"
        assert torch.all(torch.isfinite(result))
    
    def test_lnamd_memory_efficiency(self, device):
        """测试内存效率"""
        # 创建较大的特征数据
        large_features = [torch.randn(8, 512, 20, 20)]
        r_list = [1, 2]
        
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        result = LNAMD(large_features, r_list, device)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 8
        
        if torch.cuda.is_available() and device.type == "cuda":
            final_memory = torch.cuda.memory_allocated()
            # 内存增长应该是合理的
            memory_increase = final_memory - initial_memory
            assert memory_increase < 2e9  # 不应该超过2GB