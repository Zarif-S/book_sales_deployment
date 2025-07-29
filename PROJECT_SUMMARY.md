# 🧠 Project Setup Summary: ZenML + MLOps Stack (macOS)

## 🎯 Objective
Set up a full-stack, reproducible **Data Science + MLOps development workflow** on macOS with:
- Easy local development (Zed, VS Code)
- Support for Colab GPU training
- Experiment tracking (MLflow)
- Hyperparameter tuning (Optuna)
- Deployment pipelines (FastAPI + Docker + Vertex AI)
- Future MLOps integration via ZenML

## ✅ Tools Chosen
| Category              | Tool                 | Purpose |
|-----------------------|----------------------|---------|
| IDE                   | Zed + VS Code        | Local dev + notebook support |
| Env Mgmt              | Poetry + pyenv       | Python env and dependency control |
| Pipeline Orchestration | ZenML               | ML workflow orchestration |
| Experiment Tracking   | MLflow               | Track runs, models, metrics |
| Tuning                | Optuna               | Hyperparameter optimization |
| Serving               | FastAPI + Docker     | Serve model as REST API |
| Deployment Platform   | Google Vertex AI     | End-to-end cloud MLOps platform |
| Cloud Tools           | GCP SDK + Colab      | GPU training + GCP integration |

## 🧰 What You've Installed (Step 1)
- `pyenv`, `poetry`, `docker`, `colima`, `git`
- Editors: `zed`, `visual-studio-code`
- Cloud CLI: `google-cloud-sdk`

## 🔜 What's Included in This Repo
- Two ZenML pipelines:
  - 📊 `classification_pipeline/` – Classification with tuning + tracking
  - 📈 `regression_pipeline/` – Regression with tuning + tracking
- MLflow integration
- Optuna for tuning
- Colab notebooks (`colab_notebooks/`)
- FastAPI model serving (`docker/`)
- GCP deployment support (`deployment/`)
- Markdown documentation (`README.md`, `PROJECT_SUMMARY.md`)
