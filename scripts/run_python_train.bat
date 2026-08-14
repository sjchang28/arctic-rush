@echo off

REM Reduces allocator fragmentation, which is what turns a 6 GB card into an OOM.
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Run from the repository root so `src` resolves as a package.
pushd "%~dp0.."
uv run python -m src.model.train
popd
