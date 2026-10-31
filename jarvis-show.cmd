@echo off
rem jarvis-show — atalho Windows para o detector de duas palmas.
rem Coloque este .cmd ao lado de jarvis-show.py, na raiz do checkout do OpenJarvis.
setlocal
cd /d "%~dp0"
uv run python jarvis-show.py %*
