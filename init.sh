#!/bin/bash
# Environment setup (works when sourced or executed; Linux/WSL/macOS/Colab).
#
# Adapted from upstream MulTaBench: the original relied on pyenv (compiles Python from source, needs
# sudo-installed dev headers) and on `python3 -m pip` in the system interpreter (blocked by PEP 668 on
# recent Ubuntu). This version uses uv, which needs no sudo and downloads a prebuilt Python 3.11.

_multabench_init() {
    local ENV_DIR=".venv"
    local PY_VERSION="3.11"
    local VENV_PY="$ENV_DIR/bin/python"

    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "📦 uv not found. Installing uv into ~/.local/bin ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "❌ Failed to install uv"; return 1; }
    fi
    echo "🌀 Using $(uv --version)"

    if [ ! -d "$ENV_DIR" ]; then
        echo "🐍 Creating $ENV_DIR with Python $PY_VERSION (uv downloads it if missing)"
        uv venv --python "$PY_VERSION" "$ENV_DIR" || { echo "❌ Failed to create venv"; return 1; }
    else
        echo "🐍 $ENV_DIR already exists. Skipping creation."
    fi

    # Increase timeout for slow/large package downloads (default is 30s)
    export UV_HTTP_TIMEOUT=120
    echo "📄 Installing pinned dependencies from requirements.txt"
    uv pip install --python "$VENV_PY" -r requirements.txt || { echo "❌ Dependency install failed"; return 1; }

    echo "🛠 Ensuring repo root is on Python path via .pth"
    local SITE_PACKAGES
    SITE_PACKAGES=$("$VENV_PY" -c "import site; print(site.getsitepackages()[0])")
    echo "$(pwd)" > "$SITE_PACKAGES/tabstar_paper.pth"

    echo "🎉 Setup completed! To activate in a new shell, run: source $ENV_DIR/bin/activate"
}

_multabench_init "$@"
