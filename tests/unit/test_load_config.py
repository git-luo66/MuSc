"""
测试配置加载模块
"""
import pytest
import os
import yaml
import tempfile
from unittest.mock import patch, mock_open
from utils.load_config import load_yaml


class TestLoadConfig:
    """配置加载功能测试"""
    
    def test_load_yaml_success(self, temp_config_file):
        """测试成功加载YAML配置文件"""
        # 使用临时配置文件
        with patch('os.getcwd', return_value=os.path.dirname(temp_config_file)):
            config_path = os.path.basename(temp_config_file)
            result = load_yaml(config_path)
            
        assert isinstance(result, dict)
        assert 'datasets' in result
        assert 'models' in result
        assert result['datasets']['dataset_name'] == 'mvtec_ad'
    
    def test_load_yaml_file_not_found(self):
        """测试文件不存在时的错误处理"""
        with pytest.raises(FileNotFoundError):
            load_yaml('non_existent_file.yaml')
    
    def test_load_yaml_invalid_yaml(self):
        """测试无效YAML文件的错误处理"""
        invalid_yaml_content = """
        datasets:
          - invalid: yaml: content
            missing: bracket
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write(invalid_yaml_content)
            temp_file.flush()
            
            with patch('os.getcwd', return_value=os.path.dirname(temp_file.name)):
                config_path = os.path.basename(temp_file.name)
                with pytest.raises(yaml.YAMLError):
                    load_yaml(config_path)
        
        os.unlink(temp_file.name)
    
    def test_load_yaml_empty_file(self):
        """测试空文件的处理"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write("")
            temp_file.flush()
            
            with patch('os.getcwd', return_value=os.path.dirname(temp_file.name)):
                config_path = os.path.basename(temp_file.name)
                result = load_yaml(config_path)
                assert result is None
        
        os.unlink(temp_file.name)
    
    def test_load_yaml_with_different_cwd(self, temp_config_file):
        """测试在不同工作目录下加载配置文件"""
        original_dir = os.getcwd()
        config_dir = os.path.dirname(temp_config_file)
        config_name = os.path.basename(temp_config_file)
        
        # 模拟不同的工作目录
        with patch('os.getcwd', return_value='/different/path'):
            full_path = os.path.join('/different/path', config_name)
            
            # 创建完整路径的配置文件
            os.makedirs('/tmp/different/path', exist_ok=True)
            with open('/tmp/different/path/' + config_name, 'w') as f:
                yaml.dump({'test': 'data'}, f)
            
            with patch('os.path.join', return_value='/tmp/different/path/' + config_name):
                result = load_yaml(config_name)
                assert result == {'test': 'data'}
            
            # 清理
            os.remove('/tmp/different/path/' + config_name)
            os.rmdir('/tmp/different/path')