#!/bin/bash
# One-shot installer for the YAM simulation pipeline.

set -e

# Pinned IsaacLab commit (matches deps/IsaacLab; bump deliberately when upgrading).
ISAACLAB_COMMIT="f4aa17f87e2e5db5484f0b5974918573e8918ce2"

# ---- Parse args -----------------------------------------------------
ENV_NAME="yam_lab"
USE_MAMBA=0
while getopts 'e:m' flag; do
  case "${flag}" in
    e) ENV_NAME="${OPTARG}" ;;
    m) USE_MAMBA=1 ;;
    *) echo "Usage: bash install.sh [-e ENV_NAME] [-m]"; exit 1 ;;
  esac
done

if [[ "${USE_MAMBA}" == 1 ]]; then
  CONDA_CMD="mamba"
else
  CONDA_CMD="conda"
fi

# Resolve and move to the repo root (this script's directory).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
mkdir -p deps

banner() {
  printf "\n****************************************\n%s\n****************************************\n" "$1"
}

# ---- 1. Conda environment -------------------------------------------
if ${CONDA_CMD} info --envs | grep -qE "^${ENV_NAME}[[:space:]]"; then
  banner "Conda environment [${ENV_NAME}] already exists, skipping creation"
else
  banner "Creating conda environment [${ENV_NAME}] (python 3.11)..."
  ${CONDA_CMD} create -y -n "${ENV_NAME}" python=3.11
fi

# Activate the env, and terminate early if activation fails.
eval "$(conda shell.bash hook)"
if [[ "${USE_MAMBA}" == 1 ]]; then
  source "${CONDA_EXE%/bin/conda}/etc/profile.d/mamba.sh"
fi
${CONDA_CMD} activate "${ENV_NAME}"
if [[ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
  printf "Failed to activate conda environment [${ENV_NAME}] (active: [${CONDA_DEFAULT_ENV}]). Terminating.\n"
  exit 1
fi

# ---- 2. Core package ------------------------------------------------
banner "Installing the core package + PyTorch (cu128)..."
pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# ---- 3. LeRobot -----------------------------------------------------
banner "Installing LeRobot (lerobotv2.0 branch)..."
${CONDA_CMD} install -y -c conda-forge ffmpeg x264
if [ ! -d "deps/lerobot" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RogerDAI1217/lerobot.git deps/lerobot
fi
cd deps/lerobot
git checkout lerobotv2.0
pip install -e .
cd "${REPO_DIR}"

# ---- 4. Isaac Sim + IsaacLab ----------------------------------------
banner "Installing Isaac Sim 5.1.0..."
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

banner "Installing build tools (cmake, build-essential) -- requires sudo..."
sudo apt-get update && sudo apt-get install -y cmake build-essential

banner "Installing IsaacLab @ ${ISAACLAB_COMMIT}..."
if [ ! -d "deps/IsaacLab" ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git deps/IsaacLab
fi
cd deps/IsaacLab
git checkout "${ISAACLAB_COMMIT}"
# "none" skips the RL training frameworks (rl_games/rsl_rl/sb3/skrl) and robomimic
echo "Yes" | TERM=xterm ./isaaclab.sh --install none
cd "${REPO_DIR}"

# ---- 5. JoyLo -------------------------------------------------------
banner "Installing JoyLo teleoperator..."
cd joylo
pip install -e .
cd "${REPO_DIR}"

# ---- 6. Remaining dependencies --------------------------------------
banner "Installing remaining dependencies..."
${CONDA_CMD} install -y -c conda-forge "libgcc-ng>=12.3" "libstdcxx-ng>=12.3"
pip install coacd pymeshlab usd-core open3d portal

banner "Done. Activate the environment with:  conda activate ${ENV_NAME}"
