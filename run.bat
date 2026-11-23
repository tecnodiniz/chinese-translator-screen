@echo off
rem Offline Chinese -^> English screen translator.
rem Drag a rectangle around the VM window when the screen dims. Extra options are passed through.
pushd "%~dp0"
".venv\Scripts\python.exe" translator.py --select %*
popd
pause
