# IGOR

Foundation Model for Embodied AI

## New machine

```bash
# Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh

# Github
ssh-keygen -t ed25519 -C "chuhengzhang@microsoft.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_chuheng
cat ~/.ssh/id_ed25519.pub       # paste to https://github.com/settings/ssh/new  https://aka.ms/gcrssh/
git clone git@github.com:zhangchuheng123/IGOR.git -b chuheng/langat_v2
git clone git@github.com:lemmonation/igor-policy-models.git -b chuheng/sim
git clone git@github.com:msra-igor/igor-policy-finetuning.git -b real_robot
```

## Create ``openpi0`` environment

```bash
conda create -n openpi0 python=3.10 pip -y
conda activate openpi0
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y 
pip install einops gsutil==5.32 hydra-core imageio matplotlib numpy==1.26.4 omegaconf pillow pre-commit==4.0.1 pretty_errors protobuf==3.20.3 tqdm wandb h5py --user
pip install bitsandbytes==0.45.5
pip install transformers==4.47.1 tensorflow_datasets==4.9.2 tensorflow==2.15.0
pip install kornia loguru mediapy
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers triton --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
pip install timm --no-deps
pip install transforms3d packaging
pip install jsonlines mlflow tensorflow_graphics
pip install opencv-python gym

# If running server
pip install msgpack Flask
```

## Use Blobfuse2

Reference: https://learn.microsoft.com/en-us/azure/storage/blobs/blobfuse2-how-to-deploy?tabs=RHEL

```bash
cat /etc/*-release # watch ubuntu version

sudo wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install libfuse3-dev fuse3
sudo apt-get install blobfuse2
mkdir ~/blobfuse
vim ~/blobfuse/rushaml2996082614_shared.yaml
# https://github.com/Azure/azure-storage-fuse/blob/main/sampleFileCacheWithSASConfig.yaml

# Daily mount & unmount
blobfuse2 unmount all
sudo umount -l /mnt/shared_data

sudo rm -rf /mnt/shared_data; sudo rm -rf /mnt/tmp_shared
sudo mkdir /mnt/tmp_shared; sudo chmod -R 777 /mnt/tmp_shared; sudo mkdir /mnt/shared_data; sudo chmod -R 777 /mnt/shared_data
blobfuse2 mount /mnt/shared_data --config-file=~/blobfuse/rushaml2996082614_shared.yaml

sudo rm -rf /mnt/chuheng_data; sudo rm -rf /mnt/tmp_chuheng
sudo mkdir /mnt/tmp_chuheng; sudo chmod -R 777 /mnt/tmp_chuheng; sudo mkdir /mnt/chuheng_data; sudo chmod -R 777 /mnt/chuheng_data
blobfuse2 mount /mnt/chuheng_data --config-file=~/blobfuse/fastaml0167939705_chuheng.yaml
sv=2025-07-05&spr=https%2Chttp&st=2026-02-23T02%3A48%3A45Z&se=2026-03-02T02%3A48%3A00Z&skoid=51d59d7b-4235-4247-8e09-16d80993196f&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2026-02-23T02%3A48%3A45Z&ske=2026-03-02T02%3A48%3A00Z&sks=b&skv=2025-07-05&sr=c&sp=racwdxltf&sig=Zdky3r795cOd9ECJ%2BULdpU%2FaCIsH0%2Fd5a0UOO0cfxbM%3D
```

Sample Configuration
```
logging:
  type: syslog
  level: log_debug

components:
  - libfuse
  - file_cache
  - attr_cache
  - azstorage

libfuse:
  attribute-expiration-sec: 120
  entry-expiration-sec: 120
  negative-entry-expiration-sec: 240

file_cache:
  path: /mnt/tmp_shared
  timeout-sec: 120
  max-size-mb: 4096

attr_cache:
  timeout-sec: 7200

azstorage:
  type: block
  account-name: rushaml2996082614
  sas: sv=2023-01-03&spr=https%2Chttp&st=2025-04-23T05%3A43%3A33Z&se=2025-04-30T05%3A43%3A00Z&skoid=51d59d7b-4235-4247-8e09-16d80993196f&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-04-23T05%3A43%3A33Z&ske=2025-04-30T05%3A43%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=a1Y08MF2MEIcufDEcQx3JgUjZfA%2BXV2xyagRuoMLDZM%3D
  endpoint: https://rushaml2996082614.blob.core.windows.net
  mode: sas
  container: sharedcontainer
```

## Create ``amlt9`` environment
```bash
conda create -n amlt9 python=3.8 pip -y
conda activate amlt9
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
pip install amlt --index-url https://msrpypi.azurewebsites.net/stable/leloojoo --no-cache-dir
pip install azureml-k8s-mt --index-url https://msrpypi.azurewebsites.net/stable/leloojoo

az login

# checkout can be create
amlt project checkout IGOR itpseasiadata chuhengzhang
amlt project checkout IGOR drlaml1124249018 chuhengzhang
amlt project checkout IGOR lightaml3149727845 chuhengzhang
amlt project checkout IGOR bonusaml4242345487 chuhengzhang
amlt project checkout IGOR rushaml2996082614 chuhengzhang

amlt target add-defaults gcr

amlt target add --subscription 3f2ab3f5-468d-4ba7-bc14-9d3a9da4bcc5 --resource-group GCRArcK8sAMLProd2 --workspace-name gcrarck8s4ws2
amlt target add --subscription 1584df91-d540-4cd3-a9ca-24ff3dc95ba7 --resource-group UKSouth --workspace-name DRL-AML
amlt target add --subscription 1584df91-d540-4cd3-a9ca-24ff3dc95ba7 --resource-group WestUS3 --workspace-name Light-AML
amlt target add --subscription 1584df91-d540-4cd3-a9ca-24ff3dc95ba7 --resource-group JapanEast --workspace-name BonusAML 
amlt target add --subscription 1584df91-d540-4cd3-a9ca-24ff3dc95ba7 --resource-group SouthCentralUS --workspace-name RushAML 

amlt target info aml
amlt target info sing
```

## GPU (Re-)Install

```bash
sudo apt-get --purge remove "*nvidia*"; sudo reboot
sudo add-apt-repository ppa:graphics-drivers/ppa; sudo apt update; sudo apt install ubuntu-drivers-common; ubuntu-drivers devices; sudo apt install nvidia-driver-535; sudo reboot
# Try also for secure boot enabled machine
# sudo mokutil --disable-validation
# https://ubuntu.com/server/docs/nvidia-drivers-installation
# sudo ubuntu-drivers install --gpgpu nvidia:535-server
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers install nvidia:535; sudo reboot
```

## When nvidia-smi is down

```bash
sudo apt-get --purge remove "*nvidia*"; sudo reboot
sudo add-apt-repository ppa:graphics-drivers/ppa; sudo apt update; sudo apt install ubuntu-drivers-common; ubuntu-drivers devices; sudo apt install nvidia-driver-535; sudo reboot
```

## Azcopy

```bash
# sudo snap install azcli
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
sudo snap install azcli
```

```
https://rushaml2996082614.blob.core.windows.net/chuheng?sv=2023-01-03&spr=https%2Chttp&st=2025-06-06T00%3A08%3A04Z&se=2025-06-13T00%3A08%3A00Z&skoid=51d59d7b-4235-4247-8e09-16d80993196f&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-06-06T00%3A08%3A04Z&ske=2025-06-13T00%3A08%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=hW%2FkMjDP4kIkPG6rRKxuBBCjZtJrwmAL%2Bj4C9urywc0%3D
```

# Install

Each model needs its own install. Re-running install.sh with a different `--model` will reconfigure the environment.

## OpenPI (Pi-0 / Pi-0.5)

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
```

## SmolVLA

```bash
bash requirements/install.sh embodied --model smolvla --env maniskill_libero
```

# Run Pi-0.5

```bash
# Download assets
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./rlinf/envs/maniskill/assets
uv run --no-sync hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./models/RLinf-Pi05-LIBERO-SFT
bash examples/embodiment/run_libero_ppo_openpi_pi05.sh
```

# Run Pi-0 (not tested)

```bash
# Download assets
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./rlinf/envs/maniskill/assets
uv run --no-sync hf download RLinf/RLinf-Pi0-LIBERO-Long-SFT --local-dir ./models/RLinf-Pi0-LIBERO-Long-SFT
bash examples/embodiment/run_libero_ppo_openpi_pi0.sh
```

# Run SmolVLA

```bash
rm -rf .venv
bash requirements/install.sh embodied --model smolvla --env maniskill_libero
uv run --no-sync hf download HuggingFaceVLA/smolvla_libero --local-dir ./models/smolvla_libero
# Verify model config exists
test -f ./models/smolvla_libero/config.json
# Verify normalization stats exists (legacy stats.* or lerobot normalizer processor file)
bash -lc '[[ -f ./models/smolvla_libero/stats.safetensors || -f ./models/smolvla_libero/dataset_stats.safetensors || -f ./models/smolvla_libero/stats.json || -f ./models/smolvla_libero/dataset_stats.json || -n "$(compgen -G "./models/smolvla_libero/policy_preprocessor_step_*_normalizer_processor.safetensors")" ]]'
bash examples/embodiment/run_libero_ppo_smolvla.sh
```

If you see:

```text
AssertionError: `mean` is infinity. You should either initialize with `stats` as an argument, or use a pretrained model.
```

The SmolVLA checkpoint is missing or failed to load normalization stats.
Use a complete pretrained checkpoint directory (with stats files) as `MODEL_PATH`.