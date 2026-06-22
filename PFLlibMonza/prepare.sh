# Configure ~/.bashrc
echo alias p=\"ps -aux|grep zhangjq|grep 'python -u'\" >> ~/.bashrc
echo alias n=\'nvidia-smi\' >> ~/.bashrc
echo alias d=\'du -hs * | sort -h\' >> ~/.bashrc
echo alias del_pycache=\'find . -type d -name __pycache__ -prune -exec rm -rf {} \;\' >> ~/.bashrc

echo export PIP_CACHE_DIR='$PWD'/tmp >> ~/.bashrc
echo # export TMPDIR='$PWD'/tmp >> ~/.bashrc

# Install python packages from the single repo environment.
source ~/.bashrc
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
