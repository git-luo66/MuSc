# MuSc项目Makefile
# 提供便捷的开发和测试命令

.PHONY: help install test test-unit test-integration test-e2e test-gpu test-all clean lint format setup

# 默认目标
help:
	@echo "MuSc项目可用命令:"
	@echo "  setup           - 设置开发环境"
	@echo "  install         - 安装依赖"
	@echo "  test            - 运行所有测试"
	@echo "  test-unit       - 运行单元测试"
	@echo "  test-integration - 运行集成测试"
	@echo "  test-e2e        - 运行端到端测试"
	@echo "  test-gpu        - 运行GPU测试"
	@echo "  test-performance - 运行性能测试"
	@echo "  test-coverage   - 运行测试并生成覆盖率报告"
	@echo "  lint            - 运行代码检查"
	@echo "  format          - 格式化代码"
	@echo "  clean           - 清理临时文件"

# 设置开发环境
setup:
	@echo "🔧 设置开发环境..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python run_tests.py --install-deps
	pip install flake8 black isort
	@echo "✅ 开发环境设置完成"

# 安装依赖
install:
	@echo "📦 安装项目依赖..."
	pip install -r requirements.txt
	python run_tests.py --install-deps

# 运行所有测试
test:
	@echo "🧪 运行所有测试..."
	python run_tests.py --all --verbose

# 运行单元测试
test-unit:
	@echo "🔬 运行单元测试..."
	python run_tests.py --unit --verbose

# 运行集成测试
test-integration:
	@echo "🔗 运行集成测试..."
	python run_tests.py --integration --verbose

# 运行端到端测试
test-e2e:
	@echo "🎯 运行端到端测试..."
	python run_tests.py --e2e --verbose

# 运行GPU测试
test-gpu:
	@echo "🎮 运行GPU测试..."
	python run_tests.py --gpu --verbose

# 运行性能测试
test-performance:
	@echo "⚡ 运行性能测试..."
	python run_tests.py --performance --verbose

# 运行测试并生成覆盖率报告
test-coverage:
	@echo "📊 运行测试并生成覆盖率报告..."
	python run_tests.py --unit --coverage --verbose
	@echo "📈 覆盖率报告已生成在 htmlcov/ 目录"

# 运行快速测试（跳过慢测试和GPU测试）
test-fast:
	@echo "🏃 运行快速测试..."
	python run_tests.py --all --skip-slow --skip-gpu --verbose

# 代码检查
lint:
	@echo "🔍 运行代码检查..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# 代码格式化
format:
	@echo "✨ 格式化代码..."
	black .
	isort .
	@echo "✅ 代码格式化完成"

# 检查代码格式
format-check:
	@echo "🔍 检查代码格式..."
	black --check --diff .
	isort --check-only --diff .

# 清理临时文件
clean:
	@echo "🧹 清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/
	@echo "✅ 清理完成"

# 运行特定测试文件
test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ 请指定测试文件: make test-file FILE=tests/unit/test_metrics.py"; \
	else \
		echo "🧪 运行测试文件: $(FILE)"; \
		pytest $(FILE) -v; \
	fi

# 运行特定测试函数
test-func:
	@if [ -z "$(FUNC)" ]; then \
		echo "❌ 请指定测试函数: make test-func FUNC=test_compute_metrics"; \
	else \
		echo "🧪 运行测试函数: $(FUNC)"; \
		pytest -k $(FUNC) -v; \
	fi

# 生成测试报告
test-report:
	@echo "📋 生成测试报告..."
	pytest --html=test_report.html --self-contained-html
	@echo "📄 测试报告已生成: test_report.html"

# 开发模式安装
install-dev:
	@echo "🛠️  开发模式安装..."
	pip install -e .
	make setup