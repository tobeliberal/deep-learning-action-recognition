@echo off
chcp 65001 >nul
title 基于深度学习的动作识别系统
echo ========================================
echo       基于深度学习的动作识别系统
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] Python 未安装或未配置到 PATH
    pause
    exit /b 1
)

echo [信息] Python 版本:
python --version
echo.

echo [检查] 检查依赖...
python -c "import ultralytics" >nul 2>&1
if errorlevel 1 (
    echo [信息] 正在安装依赖...
    pip install -r requirements.txt
)

echo.
echo ========================================
echo       正在启动程序...
echo ========================================
echo.

python MainProgram.py

if errorlevel 1 (
    echo.
    echo [信息] 程序已退出
    pause
)
