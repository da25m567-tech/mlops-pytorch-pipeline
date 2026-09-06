# MLOps PyTorch Pipeline
# MLOps PyTorch Pipeline

An end-to-end MLOps pipeline for training and serving a **PyTorch ResNet-18 model on the CIFAR-10 dataset**, with Docker containerization and Kubernetes deployment.

## 1. Project Overview

This project demonstrates the complete machine-learning lifecycle:

* PyTorch model development and training
* CIFAR-10 dataset loading and preprocessing
* Model checkpoint generation
* Unit testing
* Dockerized model training
* Dockerized model serving
* FastAPI inference API
* Kubernetes model deployment
* Kubernetes training Job
* Kubernetes Service
* Horizontal Pod Autoscaler (HPA)
* End-to-end inference validation

### Technology Stack

| Component            | Technology     |
| -------------------- | -------------- |
| Programming Language | Python 3.11    |
| Deep Learning        | PyTorch        |
| Computer Vision      | Torchvision    |
| Dataset              | CIFAR-10       |
| Model                | ResNet-18      |
| API                  | FastAPI        |
| Containerization     | Docker         |
| Orchestration        | Kubernetes     |
| Autoscaling          | Kubernetes HPA |
| Testing              | Pytest         |

## 2. Architecture

```text
                    ┌──────────────────────┐
                    │      CIFAR-10        │
                    │       Dataset        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   PyTorch Training   │
                    │      ResNet-18       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Model Checkpoint   │
                    │ classifier_v1.pt     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Docker Container   │
                    │  Model Serving API   │
                    └──────────┬───────────┘
                               │
                               ▼
             ┌────────────────────────────────────┐
             │             Kubernetes              │
             │                                    │
             │  ┌──────────────────────────────┐  │
             │  │   FastAPI Serving Deployment │  │
             │  │        ResNet-18             │  │
             │  └──────────────┬───────────────┘  │
             │                 │                  │
             │                 ▼                  │
             │  ┌──────────────────────────────┐  │
             │  │     Kubernetes Service       │  │
             │  │          Port 8000           │  │
             │  └──────────────┬───────────────┘  │
             │                 │                  │
             │                 ▼                  │
             │          /predict API              │
             │                                    │
             │  ┌──────────────────────────────┐  │
             │  │       HPA: 1–3 replicas      │  │
             │  │       CPU target: 70%        │  │
             │  └──────────────────────────────┘  │
             └────────────────────────────────────┘
```

## 3. Repository Structure

```text
mlops-pytorch-pipeline/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── serve.py
│
├── configs/
│   └── training_config.yaml
│
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
│
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
│
├── requirements/
│   ├── train.txt
│   └── serve.txt
│
├── tests/
│   └── test_model.py
│
├── Dockerfile.train
└── README.md
```

## 4. Local Setup

Clone the repository:

```bash
git clone https://github.com/da25m567-tech/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
```

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install training dependencies:

```powershell
pip install -r requirements/train.txt
```

Install serving dependencies:

```powershell
pip install -r requirements/serve.txt
```

## 5. Run Tests

On Windows PowerShell, set the project directory as the Python path:

```powershell
$env:PYTHONPATH = (Get-Location).Path
pytest -q
```

Expected result:

```text
2 passed
```

## 6. Docker

### Build the Training Image

```powershell
docker build -f docker/Dockerfile.train -t mlops-pytorch-train:latest .
```

### Build the Serving Image

```powershell
docker build -f docker/Dockerfile.serve -t mlops-pytorch-serve:latest .
```

The Docker images package the Python environment and application code so that the pipeline can run consistently in containers.

## 7. Kubernetes Deployment

The Kubernetes manifests are located in the `k8s/` directory.

Create the namespace:

```powershell
kubectl apply -f k8s/namespace.yaml
```

Apply configuration:

```powershell
kubectl apply -f k8s/configmap.yaml
```

Deploy the model-serving application:

```powershell
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
```

Apply the autoscaler:

```powershell
kubectl apply -f k8s/hpa.yaml
```

Run the training Job:

```powershell
kubectl apply -f k8s/training-job.yaml
```

Check the deployment:

```powershell
kubectl get pods,svc,deploy,hpa -n mlops
```

## 8. Model Serving API

The FastAPI service provides the following endpoints.

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "device": "cpu"
}
```

### Model Information

```text
GET /model-info
```

Returns the model architecture, number of classes, and CIFAR-10 class names.

### Prediction

```text
POST /predict
```

The endpoint accepts an image file and returns the predicted CIFAR-10 class and confidence.

Example:

```text
class_index: 2
class_name: bird
confidence: 0.3509
```

## 9. Kubernetes Validation

Check the training Job:

```powershell
kubectl get job mlops-training -n mlops
```

The completed training Job produced:

```text
STATUS: Complete
COMPLETIONS: 1/1
DURATION: 22m
```

Training logs showed:

```text
training_started
device: cpu
epoch: 1
checkpoint_saved
training_complete
```

Check the serving deployment:

```powershell
kubectl get deployment -n mlops
```

Check the Service:

```powershell
kubectl get service -n mlops
```

Check the HPA:

```powershell
kubectl get hpa -n mlops
```

The configured HPA scales the serving deployment between **1 and 3 replicas**, with a CPU utilization target of **70%**.

## 10. End-to-End Inference Validation

Port-forward the Kubernetes Service:

```powershell
kubectl port-forward -n mlops svc/mlops-serving 8001:8000
```

In another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

Then:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/model-info
```

Test image prediction:

```powershell
curl.exe -X POST "http://127.0.0.1:8001/predict" -F "file=@C:\path\to\image.png"
```

The deployed API successfully returned a CIFAR-10 prediction with a class index, class name, and confidence score.

## 11. Git Workflow

The project was developed using feature branches and pull requests.

Major development stages included:

1. PyTorch model and training pipeline
2. Dockerized training
3. Dockerized model serving
4. Kubernetes deployment and autoscaling

The final Kubernetes implementation was merged through **Pull Request #4**.

The final repository contains the merged implementation on the `main` branch.

## 12. Final Outcome

The project demonstrates an end-to-end PyTorch MLOps workflow:

```text
Dataset
   ↓
PyTorch Training
   ↓
Model Checkpoint
   ↓
Docker
   ↓
Kubernetes
   ↓
FastAPI Model Serving
   ↓
Prediction
   ↓
HPA-based Autoscaling
```

The pipeline was validated through unit tests, Kubernetes workload checks, API health checks, model-information checks, and an end-to-end prediction request.
