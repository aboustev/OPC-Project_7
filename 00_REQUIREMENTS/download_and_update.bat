@echo off
cd C:\Users\aboudiwan
pip install -r requirements.txt
timeout 3
pip install -r requirements.txt --upgrade
echo 'work done'
timeout 3