#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates pkg-config \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libncurses-dev xz-utils tk-dev libffi-dev liblzma-dev libzstd-dev \
  cmake libboost-all-dev libeigen3-dev nlohmann-json3-dev libjsoncpp-dev ffmpeg

export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
if [[ ! -d "$PYENV_ROOT" ]]; then
  git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
fi
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

python3 - <<'PY'
from pathlib import Path
block = '''# >>> SpatialPreferential pyenv >>>
export PYENV_ROOT="$HOME/.pyenv"
[[ -d "$PYENV_ROOT/bin" ]] && export PATH="$PYENV_ROOT/bin:$PATH"
command -v pyenv >/dev/null 2>&1 && eval "$(pyenv init - bash)"
# <<< SpatialPreferential pyenv <<<
'''
for name in ('.bashrc', '.profile'):
    path = Path.home() / name
    content = path.read_text() if path.exists() else ''
    if '# >>> SpatialPreferential pyenv >>>' not in content:
        with path.open('a') as stream:
            stream.write('\n' + block)
PY

cd "$PROJECT_ROOT"
if [[ -f .python-version ]]; then
  PYTHON_VERSION="$(cat .python-version)"
else
  PYTHON_VERSION="$(pyenv latest -k 3.12)"
fi
export MAKE_OPTS="-j2"
pyenv install -s "$PYTHON_VERSION"
pyenv local "$PYTHON_VERSION"
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > requirements.lock.txt
.venv/bin/python -m ipykernel install --user --name spatialpreferential \
  --display-name "Python (SpatialPreferential)"
.venv/bin/python -m pip check
echo 'Activate with: source .venv/bin/activate'
