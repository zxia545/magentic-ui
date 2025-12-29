
# if .venv does not exist, create it
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate


pip install -e .

