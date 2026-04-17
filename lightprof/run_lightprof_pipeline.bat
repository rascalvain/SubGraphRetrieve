@echo off
REM LightPROF 完整Pipeline批处理脚本
REM 使用方法: run_lightprof_pipeline.bat <checkpoint_path>

setlocal enabledelayedexpansion

REM 检查参数
if "%~1"=="" (
    echo 错误: 请提供检查点路径
    echo 使用方法: run_lightprof_pipeline.bat webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth
    exit /b 1
)

set CHECKPOINT=%~1
set CHECKPOINT_DIR=%~dp1

echo ================================================================================
echo LightPROF 完整Pipeline
echo ================================================================================
echo 检查点: %CHECKPOINT%
echo 输出目录: %CHECKPOINT_DIR%
echo ================================================================================
echo.

REM 激活conda环境（根据实际情况修改）
call conda activate retriever
if errorlevel 1 (
    echo 警告: conda环境激活失败，尝试继续...
)

REM 步骤1: Retriever推理（如果结果不存在）
if not exist "%CHECKPOINT_DIR%val_retrieval_result.pth" (
    echo [步骤1/3] 运行Retriever推理...
    python inference_hybrid.py ^
        -p %CHECKPOINT% ^
        --splits val test ^
        --enable_entity_mapping ^
        --num_display_samples 0
    
    if errorlevel 1 (
        echo 错误: Retriever推理失败
        exit /b 1
    )
    echo.
) else (
    echo [步骤1/3] 检索结果已存在，跳过...
    echo.
)

REM 步骤2a: LightPROF采样（验证集）
echo [步骤2/3] 对验证集应用LightPROF采样...
python lightprof_sampling.py ^
    -i "%CHECKPOINT_DIR%val_retrieval_result.pth" ^
    --h_q 2 ^
    --top_k_chains 5 ^
    --max_paths_per_chain 10 ^
    --use_mock_llm ^
    --num_display_samples 2

if errorlevel 1 (
    echo 错误: 验证集采样失败
    exit /b 1
)
echo.

REM 步骤2b: LightPROF采样（测试集）
echo [步骤3/3] 对测试集应用LightPROF采样...
python lightprof_sampling.py ^
    -i "%CHECKPOINT_DIR%test_retrieval_result.pth" ^
    --h_q 2 ^
    --top_k_chains 5 ^
    --max_paths_per_chain 10 ^
    --use_mock_llm ^
    --num_display_samples 2

if errorlevel 1 (
    echo 错误: 测试集采样失败
    exit /b 1
)
echo.

echo ================================================================================
echo 完成！生成的文件：
echo ================================================================================
echo 验证集:
echo   - %CHECKPOINT_DIR%val_retrieval_result_lightprof.pth
echo   - %CHECKPOINT_DIR%val_retrieval_result_lightprof_stats.json
echo.
echo 测试集:
echo   - %CHECKPOINT_DIR%test_retrieval_result_lightprof.pth
echo   - %CHECKPOINT_DIR%test_retrieval_result_lightprof_stats.json
echo ================================================================================
echo.

pause
