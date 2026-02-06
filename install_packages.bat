@echo off
echo Installing required packages...
echo.

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy torch matplotlib gymnasium pygame-ce

echo.
echo Installation complete!
pause
