#!/bin/bash
# MuSc项目测试运行脚本

set -e

echo "🚀 MuSc项目测试套件"
echo "===================="

# 检查Python环境
python --version
echo ""

# 检查是否安装了pytest
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest未安装，正在安装测试依赖..."
    python run_tests.py --install-deps
fi

# 解析命令行参数
VERBOSE=""
COVERAGE=""
SKIP_SLOW=""
SKIP_GPU=""
TEST_TYPE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --coverage)
            COVERAGE="--coverage"
            shift
            ;;
        --skip-slow)
            SKIP_SLOW="--skip-slow"
            shift
            ;;
        --skip-gpu)
            SKIP_GPU="--skip-gpu"
            shift
            ;;
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --e2e)
            TEST_TYPE="e2e"
            shift
            ;;
        --gpu)
            TEST_TYPE="gpu"
            shift
            ;;
        --performance)
            TEST_TYPE="performance"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -v, --verbose     详细输出"
            echo "  --coverage        生成覆盖率报告"
            echo "  --skip-slow       跳过慢测试"
            echo "  --skip-gpu        跳过GPU测试"
            echo "  --unit            只运行单元测试"
            echo "  --integration     只运行集成测试"
            echo "  --e2e             只运行端到端测试"
            echo "  --gpu             只运行GPU测试"
            echo "  --performance     只运行性能测试"
            echo "  -h, --help        显示帮助信息"
            exit 0
            ;;
        *)
            echo "❌ 未知选项: $1"
            echo "使用 $0 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python run_tests.py"

if [ "$TEST_TYPE" != "all" ]; then
    CMD="$CMD --$TEST_TYPE"
else
    CMD="$CMD --all"
fi

if [ -n "$VERBOSE" ]; then
    CMD="$CMD $VERBOSE"
fi

if [ -n "$COVERAGE" ]; then
    CMD="$CMD $COVERAGE"
fi

if [ -n "$SKIP_SLOW" ]; then
    CMD="$CMD $SKIP_SLOW"
fi

if [ -n "$SKIP_GPU" ]; then
    CMD="$CMD $SKIP_GPU"
fi

# 运行测试
echo "🏃 执行命令: $CMD"
echo ""

exec $CMD