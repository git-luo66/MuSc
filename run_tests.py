#!/usr/bin/env python3
"""
测试运行脚本
提供便捷的测试执行接口
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, cwd=None):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def install_test_dependencies():
    """安装测试依赖"""
    print("🔧 正在安装测试依赖...")
    
    test_requirements = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "pytest-xdist>=3.0.0",  # 并行测试
        "pytest-timeout>=2.1.0",  # 超时控制
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scikit-image>=0.19.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0"
    ]
    
    for requirement in test_requirements:
        returncode, stdout, stderr = run_command(f"pip install {requirement}")
        if returncode != 0:
            print(f"❌ 安装 {requirement} 失败: {stderr}")
            return False
        else:
            print(f"✅ 已安装 {requirement}")
    
    return True


def run_unit_tests(verbose=False, coverage=False):
    """运行单元测试"""
    print("🧪 运行单元测试...")
    
    cmd = "pytest tests/unit/"
    if verbose:
        cmd += " -v"
    if coverage:
        cmd += " --cov=utils --cov=models --cov-report=html --cov-report=term"
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print("✅ 单元测试通过")
        print(stdout)
    else:
        print("❌ 单元测试失败")
        print(stderr)
    
    return returncode == 0


def run_integration_tests(verbose=False):
    """运行集成测试"""
    print("🔗 运行集成测试...")
    
    cmd = "pytest tests/integration/"
    if verbose:
        cmd += " -v"
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print("✅ 集成测试通过")
        print(stdout)
    else:
        print("❌ 集成测试失败")
        print(stderr)
    
    return returncode == 0


def run_e2e_tests(verbose=False):
    """运行端到端测试"""
    print("🎯 运行端到端测试...")
    
    cmd = "pytest tests/test_e2e.py"
    if verbose:
        cmd += " -v"
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print("✅ 端到端测试通过")
        print(stdout)
    else:
        print("❌ 端到端测试失败")
        print(stderr)
    
    return returncode == 0


def run_performance_tests():
    """运行性能测试"""
    print("⚡ 运行性能测试...")
    
    cmd = "pytest -m slow --timeout=300"
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print("✅ 性能测试通过")
        print(stdout)
    else:
        print("❌ 性能测试失败")
        print(stderr)
    
    return returncode == 0


def run_gpu_tests():
    """运行GPU测试"""
    if not torch.cuda.is_available():
        print("⚠️  跳过GPU测试（未检测到CUDA）")
        return True
    
    print("🎮 运行GPU测试...")
    
    cmd = "pytest -m gpu"
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print("✅ GPU测试通过")
        print(stdout)
    else:
        print("❌ GPU测试失败")
        print(stderr)
    
    return returncode == 0


def run_all_tests(verbose=False, coverage=False, skip_slow=False, skip_gpu=False):
    """运行所有测试"""
    print("🚀 开始运行完整测试套件...")
    
    results = []
    
    # 运行单元测试
    results.append(("单元测试", run_unit_tests(verbose, coverage)))
    
    # 运行集成测试
    results.append(("集成测试", run_integration_tests(verbose)))
    
    # 运行端到端测试
    results.append(("端到端测试", run_e2e_tests(verbose)))
    
    # 运行性能测试
    if not skip_slow:
        results.append(("性能测试", run_performance_tests()))
    
    # 运行GPU测试
    if not skip_gpu:
        results.append(("GPU测试", run_gpu_tests()))
    
    # 汇总结果
    print("\n📊 测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("=" * 50)
    print(f"总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试都通过了！")
        return True
    else:
        print("💥 有测试失败，请检查错误信息")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MuSc项目测试运行器")
    parser.add_argument("--install-deps", action="store_true", help="安装测试依赖")
    parser.add_argument("--unit", action="store_true", help="只运行单元测试")
    parser.add_argument("--integration", action="store_true", help="只运行集成测试")
    parser.add_argument("--e2e", action="store_true", help="只运行端到端测试")
    parser.add_argument("--performance", action="store_true", help="只运行性能测试")
    parser.add_argument("--gpu", action="store_true", help="只运行GPU测试")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--coverage", action="store_true", help="生成覆盖率报告")
    parser.add_argument("--skip-slow", action="store_true", help="跳过慢测试")
    parser.add_argument("--skip-gpu", action="store_true", help="跳过GPU测试")
    
    args = parser.parse_args()
    
    # 安装依赖
    if args.install_deps:
        if not install_test_dependencies():
            sys.exit(1)
        return
    
    # 检查pytest是否可用
    try:
        import pytest
    except ImportError:
        print("❌ pytest未安装，请先运行: python run_tests.py --install-deps")
        sys.exit(1)
    
    # 根据参数运行相应的测试
    success = True
    
    if args.unit:
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.integration:
        success = run_integration_tests(args.verbose)
    elif args.e2e:
        success = run_e2e_tests(args.verbose)
    elif args.performance:
        success = run_performance_tests()
    elif args.gpu:
        success = run_gpu_tests()
    elif args.all or not any([args.unit, args.integration, args.e2e, args.performance, args.gpu]):
        success = run_all_tests(args.verbose, args.coverage, args.skip_slow, args.skip_gpu)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()