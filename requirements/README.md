## Dependency Installation Guide

We recommend using [`uv`](https://docs.astral.sh/uv/) to install the necessary Python dependencies.
You can install `uv` via `pip`.
```shell
pip install --upgrade uv
```

After installing `uv`, you can install the dependencies for the target experiments using the `install.sh` script under the `requirements/` folder.
The script is organized by **targets** and **models**:

- `embodied` target (embodied agents) with models:
	- `openvla`
	- `openvla-oft`
	- `openpi`

	Each embodied model also requires an `--env` argument to specify the environment, e.g. `maniskill_libero`, `behavior`, or `metaworld`.

- `reason` target (reasoning / Megatron stack).

For example, to install the dependencies for the OpenVLA + ManiSkill LIBERO experiment, you would run:
```shell
bash requirements/install.sh embodied --model openvla --env maniskill_libero
```

This will create a virtual environment under the current path named `.venv`.
To activate the virtual environment, you can use the following command:
```shell
source .venv/bin/activate
```

To deactivate the virtual environment, simply run:
```shell
deactivate
```

For `maniskill_libero`, `install.sh` now performs post-install environment setup automatically to reduce runtime warning spam:

- runs `robosuite/scripts/setup_macros.py` (creates `macros_private.py`)
- installs `PyOpenGL-accelerate`
- runs a quick import-based post-check

Optional environment switches before installation:

```shell
# Default: 1 (run robosuite macro setup)
export MANISKILL_LIBERO_SETUP_MACROS=1

# Default: 1 (install OpenGL acceleration package)
export MANISKILL_LIBERO_INSTALL_OPENGL_ACCELERATE=1

# Default: 0 (keep gym-notices installed)
export MANISKILL_LIBERO_SILENCE_GYM_NOTICE=0
```

To install the reasoning (Megatron + SGLang/vLLM) stack instead, run:
```shell
bash requirements/install.sh agentic
```

You can override the default virtual environment directory using `--venv`. For example:
```shell
bash requirements/install.sh embodied --model openpi --env maniskill_libero --venv openpi-venv
```