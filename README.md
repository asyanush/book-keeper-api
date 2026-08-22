# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install pip-tools
python -m pip install pip-tools

# 5. Compile requirements-dev.in
pip-compile requirements-dev.in

# 6. Install requirements
pip install -r requirements-dev.txt