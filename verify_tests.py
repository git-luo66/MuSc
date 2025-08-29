#!/usr/bin/env python3
"""
测试代码验证脚本
验证测试文件的语法和结构正确性
"""
import os
import ast
import sys
from pathlib import Path


def check_python_syntax(file_path):
    """检查Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析AST检查语法
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"


def analyze_test_file(file_path):
    """分析测试文件结构"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return {
            'classes': classes,
            'functions': functions,
            'imports': imports[:10],  # 只显示前10个导入
            'test_functions': [f for f in functions if f.startswith('test_')]
        }
    except Exception as e:
        return {'error': str(e)}


def verify_test_structure():
    """验证测试结构"""
    print("🔍 验证测试代码结构...")
    print("=" * 60)
    
    test_dir = Path('/workspace/tests')
    if not test_dir.exists():
        print("❌ 测试目录不存在")
        return False
    
    # 检查主要测试文件
    test_files = [
        'tests/conftest.py',
        'tests/unit/test_load_config.py',
        'tests/unit/test_metrics.py',
        'tests/unit/test_msm.py',
        'tests/unit/test_rscin.py',
        'tests/unit/test_lnamd.py',
        'tests/integration/test_musc_integration.py',
        'tests/data/mock_data.py',
        'tests/test_e2e.py'
    ]
    
    all_valid = True
    
    for test_file in test_files:
        file_path = Path('/workspace') / test_file
        
        if not file_path.exists():
            print(f"❌ 文件不存在: {test_file}")
            all_valid = False
            continue
        
        # 检查语法
        is_valid, error = check_python_syntax(file_path)
        if not is_valid:
            print(f"❌ {test_file}: {error}")
            all_valid = False
            continue
        
        # 分析文件结构
        analysis = analyze_test_file(file_path)
        if 'error' in analysis:
            print(f"❌ {test_file}: {analysis['error']}")
            all_valid = False
            continue
        
        print(f"✅ {test_file}")
        print(f"   📦 类: {len(analysis['classes'])} 个")
        print(f"   🔧 函数: {len(analysis['functions'])} 个")
        print(f"   🧪 测试函数: {len(analysis['test_functions'])} 个")
        
        if analysis['test_functions']:
            print(f"   📝 测试方法示例: {', '.join(analysis['test_functions'][:3])}")
        print()
    
    return all_valid


def verify_config_files():
    """验证配置文件"""
    print("🔧 验证配置文件...")
    print("=" * 60)
    
    config_files = [
        ('pytest.ini', 'pytest配置'),
        ('Makefile', 'Make构建配置'),
        ('test-requirements.txt', '测试依赖'),
        ('.github/workflows/test.yml', 'GitHub Actions配置'),
        ('run_tests.py', '测试运行器'),
        ('test.sh', 'Bash测试脚本')
    ]
    
    all_exist = True
    
    for file_path, description in config_files:
        full_path = Path('/workspace') / file_path
        if full_path.exists():
            file_size = full_path.stat().st_size
            print(f"✅ {description}: {file_path} ({file_size} bytes)")
        else:
            print(f"❌ {description}: {file_path} 不存在")
            all_exist = False
    
    return all_exist


def count_test_coverage():
    """统计测试覆盖范围"""
    print("📊 测试覆盖统计...")
    print("=" * 60)
    
    # 统计测试文件
    test_files = list(Path('/workspace/tests').rglob('test_*.py'))
    unit_tests = list(Path('/workspace/tests/unit').glob('test_*.py'))
    integration_tests = list(Path('/workspace/tests/integration').glob('test_*.py'))
    
    print(f"📁 总测试文件: {len(test_files)} 个")
    print(f"🔬 单元测试文件: {len(unit_tests)} 个")
    print(f"🔗 集成测试文件: {len(integration_tests)} 个")
    print()
    
    # 统计测试函数数量
    total_test_functions = 0
    
    for test_file in test_files:
        if test_file.name.startswith('test_'):
            analysis = analyze_test_file(test_file)
            if 'test_functions' in analysis:
                test_count = len(analysis['test_functions'])
                total_test_functions += test_count
                print(f"  📝 {test_file.name}: {test_count} 个测试函数")
    
    print()
    print(f"🎯 总测试函数: {total_test_functions} 个")
    
    return total_test_functions


def verify_project_structure():
    """验证项目结构完整性"""
    print("🏗️  验证项目结构...")
    print("=" * 60)
    
    # 检查原始项目文件
    original_files = [
        'examples/musc_main.py',
        'models/musc.py',
        'utils/metrics.py',
        'utils/load_config.py',
        'configs/musc.yaml',
        'requirements.txt'
    ]
    
    print("📋 原始项目文件:")
    for file_path in original_files:
        full_path = Path('/workspace') / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} 缺失")
    
    print()
    
    # 检查新增测试文件
    test_files = [
        'tests/conftest.py',
        'tests/unit/test_metrics.py',
        'tests/integration/test_musc_integration.py',
        'run_tests.py',
        'pytest.ini'
    ]
    
    print("🧪 新增测试文件:")
    for file_path in test_files:
        full_path = Path('/workspace') / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} 缺失")


def main():
    """主验证函数"""
    print("🎯 MuSc项目测试代码验证")
    print("=" * 60)
    print()
    
    # 验证项目结构
    verify_project_structure()
    print()
    
    # 验证测试文件
    test_structure_valid = verify_test_structure()
    print()
    
    # 验证配置文件
    config_valid = verify_config_files()
    print()
    
    # 统计覆盖范围
    test_count = count_test_coverage()
    print()
    
    # 总结
    print("📋 验证总结")
    print("=" * 60)
    
    if test_structure_valid:
        print("✅ 测试文件结构正确")
    else:
        print("❌ 测试文件存在问题")
    
    if config_valid:
        print("✅ 配置文件完整")
    else:
        print("❌ 配置文件缺失")
    
    print(f"📊 生成了 {test_count} 个测试函数")
    
    print()
    print("🎉 测试代码生成完成！")
    print()
    print("📚 使用说明:")
    print("  1. 安装依赖: python3 run_tests.py --install-deps")
    print("  2. 运行测试: python3 run_tests.py --all")
    print("  3. 查看文档: cat tests/README.md")
    print()
    print("🚀 快速命令:")
    print("  make setup    # 设置环境")
    print("  make test     # 运行测试")
    print("  make help     # 查看所有命令")


if __name__ == "__main__":
    main()