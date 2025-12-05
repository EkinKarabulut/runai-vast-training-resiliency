# Run:ai & VAST Data — Resilient Training Demo

This repository demonstrates how to combine NVIDIA Run:ai (for GPU orchestration and AI workload scheduling) with VAST Data (for fast, secure checkpoint storage) to build a training setup that automatically recovers from failures.

## 🚧 Demo Overview

1. **Train** a LoRa adapter on Llama 3.1  
2. **Checkpoint** every **5 training steps** to a VAST‑mounted volume  
3. **Simulate a GPU host crash** in `us-east-1`  
4. **Run:ai** detects the failure and auto‑reschedules the job to `us-west-2`  
5. Training resumes seamlessly from the latest checkpoint

## 📁 Repository Structure

| Path                          | Description                                                        |
|-------------------------------|--------------------------------------------------------------------|
| **`training/`**               | Training logic, container and launch scripts                       |
| `training/distributed.py`     | Training script with checkpointing every 5 steps                   |
| `training/Dockerfile`         | Builds the trainer image and invokes `launch.sh`                   |
| `training/requirements.txt`   | Python dependencies for training                                   |
| `training/launch.sh`          | Entrypoint: calls `torchrun` for `distributed.py`                  |
| **`model-loading/`**          | Model‑loading logic to VAST storage                  |
| `model-loading/clone.sh`      | Bash script to fetch the base model & initialize checkpoint dir    |

### 🔍 Training Details (`distributed.py`)
In `distributed.py`:

- Model path: `/model/Meta-Llama-3.1-8B-Instruct`
- Checkpoint path: `/model/checkpoints/Meta-Llama-3.1-8B-Instruct-finetuned`
- Checkpoint frequency: every 5 training steps
- Resume logic: automatically finds and loads the latest checkpoint
- Batch size: 1 per device

## ✅ Prerequisites

Before you begin, make sure you have the following ready:

- ✅ A **Kubernetes cluster** 
- ✅ **Run:ai** installed in the cluster
- ✅ A **VAST Data namespace** provisioned and mounted in Kubernetes as a `PersistentVolume` (PV) and `PersistentVolumeClaim` (PVC)  
- ✅ Sufficient GPU resources (L40S/A100/H100 or equivalent)  
- ✅ Docker + a container registry to push training and model-prep images

## 🛠️ Demo Flow - Step-by-step
### 1. Clone the repository

```
git clone https://github.com/EkinKarabulut/runai-vast-training-resiliency.git
```
### 2. Provision VAST Data
Create a PVC with VAST CSI filesystem and mount it to `/model`.

### 3. Load the Model

- Launch a Docker container using the PyTorch image via the Run:ai dashboard.
- Exec into the container using:
 `kubectl exec -n $PROJECT_NAME -it $POD_NAME -- /bin/bash`
- Run the `clone.sh` script to download and prepare the model. This pulls the model from Hugging Face Hub and prepares the checkpoint directory inside your mounted VAST volume. This writes the model to /model/Meta-Llama-3.1-8B-Instruct and creates /model/checkpoints with correct permissions.

> **Note:** The VAST volume will be mounted into `/model` and should provide two directories:
1. `/model/Meta-Llama-3.1-8B-Instruct` — the base model  
2. `/model/checkpoints` — where training checkpoints are written

### 4. Build & Push Training Image

This image installs dependencies (`requirements.txt`) and includes:

- `distributed.py` — fine‑tunes Llama 3.1 LoRa adapter
- Checkpoint logic saving to `/model/checkpoints` every 5 steps

```bash
cd ../training/
docker build --platform linux/amd64 -t <REGISTRY>/runai-vast-training .
docker push <REGISTRY>/training:latest
```
Here is [the link to](https://hub.docker.com/r/ekarabulut844/runai-vast-training) the training Docker image that we built for this demo.

### 5. Launch the Training Job

Once the model is in place, launch your training job:
```
runai training pytorch submit lora-fine-tune-model --existing-pvc "claimname=$DATA,path=/model"  -i docker.io/<REGISTRY>/runai-vast-training -g 1
```

### 🧠 What to Expect During Training
Once the training job starts, logs will show the LoRA fine‑tuning in progress. Every 5 steps, a checkpoint will be saved:
```
Checkpointing to the directory ./checkpoints/Meta-Llama-3.1-8B-Instruct-finetuned/checkpoint-xxx
```
If the pod crashes, is preempted, or intentionally restarted, the next training run will automatically resume from the latest checkpoint:
```
Resuming training from latest checkpoint: ./checkpoints/Meta-Llama-3.1-8B-Instruct-finetuned/checkpoint-xxx
```
