# MuSc 测试文档

这个目录包含了MuSc项目的完整测试套件。

## 📁 测试结构

```
tests/
├── __init__.py              # 测试包初始化
├── conftest.py             # pytest配置和fixtures
├── README.md               # 测试文档（本文件）
├── unit/                   # 单元测试
│   ├── test_load_config.py # 配置加载测试
│   ├── test_metrics.py     # 评估指标测试
│   ├── test_msm.py         # MSM模块测试
│   ├── test_rscin.py       # RsCIN模块测试
│   └── test_lnamd.py       # LNAMD模块测试
├── integration/            # 集成测试
│   └── test_musc_integration.py # MuSc模型集成测试
├── data/                   # 测试数据和Mock对象
│   ├── mock_data.py        # 模拟数据生成器
│   └── __init__.py
└── test_e2e.py            # 端到端测试
```

## 🚀 快速开始

### 1. 安装测试依赖

```bash
# 使用脚本安装
python run_tests.py --install-deps

# 或者使用Makefile
make setup

# 或者手动安装
pip install -r test-requirements.txt
```

### 2. 运行测试

```bash
# 运行所有测试
python run_tests.py --all

# 或者使用Makefile
make test

# 运行特定类型的测试
make test-unit           # 单元测试
make test-integration    # 集成测试
make test-e2e           # 端到端测试
make test-gpu           # GPU测试
make test-performance   # 性能测试
```

### 3. 生成覆盖率报告

```bash
# 生成HTML覆盖率报告
make test-coverage

# 查看报告
open htmlcov/index.html
```

## 🧪 测试类型

### 单元测试 (Unit Tests)
- **位置**: `tests/unit/`
- **目的**: 测试单个函数和类的功能
- **特点**: 快速、独立、使用Mock对象
- **覆盖模块**:
  - `utils.load_config` - 配置文件加载
  - `utils.metrics` - 评估指标计算
  - `models.modules._MSM` - 互评分模块
  - `models.modules._RsCIN` - 关系引导评分
  - `models.modules._LNAMD` - 局部邻域感知异常检测

### 集成测试 (Integration Tests)
- **位置**: `tests/integration/`
- **目的**: 测试模块间的交互
- **特点**: 测试组件集成、数据流
- **覆盖功能**:
  - MuSc模型初始化
  - 数据集加载流程
  - 可视化功能
  - 设备处理

### 端到端测试 (E2E Tests)
- **位置**: `tests/test_e2e.py`
- **目的**: 测试完整工作流程
- **特点**: 模拟真实使用场景
- **覆盖流程**:
  - 配置文件到模型初始化
  - 数据加载到结果输出
  - 错误处理和边界情况

## 🏷️ 测试标记

使用pytest标记来分类和选择测试：

```bash
# 运行特定标记的测试
pytest -m unit          # 单元测试
pytest -m integration   # 集成测试
pytest -m slow          # 慢测试
pytest -m gpu           # GPU测试

# 跳过特定标记
pytest -m "not slow"    # 跳过慢测试
pytest -m "not gpu"     # 跳过GPU测试
```

## 📊 测试覆盖率

项目目标测试覆盖率：
- **单元测试**: > 90%
- **集成测试**: > 80%
- **整体覆盖率**: > 85%

查看覆盖率报告：
```bash
make test-coverage
open htmlcov/index.html
```

## 🔧 开发测试

### 运行单个测试文件
```bash
make test-file FILE=tests/unit/test_metrics.py
```

### 运行特定测试函数
```bash
make test-func FUNC=test_compute_metrics
```

### 调试模式
```bash
# 在失败时进入调试器
pytest --pdb

# 显示详细输出
pytest -s -v

# 只运行失败的测试
pytest --lf
```

## 🎯 性能测试

性能测试关注：
- **内存使用**: 确保没有内存泄漏
- **计算效率**: 验证算法性能
- **GPU利用率**: 检查GPU内存使用

```bash
# 运行性能测试
make test-performance

# 使用内存分析
pytest --memray tests/unit/test_msm.py::TestMSM::test_msm_memory_efficiency
```

## 🐛 常见问题

### Q: 测试运行很慢
A: 使用快速测试模式跳过慢测试：
```bash
make test-fast
```

### Q: GPU测试失败
A: 确保：
1. 安装了CUDA
2. PyTorch支持CUDA
3. 有足够的GPU内存

### Q: 依赖安装失败
A: 尝试：
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Q: 测试数据问题
A: 测试使用模拟数据，不需要真实数据集。如果有问题，检查`tests/data/mock_data.py`。

## 📝 编写新测试

### 单元测试模板
```python
import pytest
from your_module import your_function

class TestYourFunction:
    def test_basic_functionality(self):
        result = your_function(input_data)
        assert result == expected_output
    
    def test_edge_cases(self):
        # 测试边界情况
        pass
    
    def test_error_handling(self):
        with pytest.raises(ExpectedError):
            your_function(invalid_input)
```

### 集成测试模板
```python
@patch('external_dependency')
def test_integration_workflow(self, mock_dependency):
    # 设置模拟
    mock_dependency.return_value = mock_result
    
    # 执行集成流程
    result = integration_function()
    
    # 验证结果
    assert result.success
```

## 🔄 持续集成

项目使用GitHub Actions进行自动化测试：
- **推送到main/develop分支**: 运行完整测试套件
- **Pull Request**: 运行快速测试
- **GPU标签**: 触发GPU测试

查看CI配置：`.github/workflows/test.yml`