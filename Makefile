# 个贷违约预测 —— 常用命令（在项目根目录执行）
# 环境：WSL Ubuntu-24.04 + .venv（python3.12）

PY := .venv/bin/python

.PHONY: setup verify test clean help

help:
	@echo "make setup   创建 .venv 并安装 requirements.txt"
	@echo "make verify  数据自检（MD5/行列数/提交样例对齐）"
	@echo "make test    跑单测"
	@echo "make clean   清缓存"

setup:
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

verify:
	$(PY) scripts/00_verify_data.py

test:
	$(PY) -m pytest

clean:
	rm -rf .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
