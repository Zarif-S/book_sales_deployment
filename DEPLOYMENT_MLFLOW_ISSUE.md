# MLflow Production Deployment Issue

## 🎯 Simplified Solution for vertex_stack Users

**Status**: ✅ **SIMPLIFIED SOLUTION** - Much easier fix available with existing vertex_stack

**Last Updated**: August 8, 2025

---

## vertex_stack Analysis

**GOOD NEWS**: Your existing `vertex_stack` infrastructure already solves most deployment issues!

### Current Infrastructure
```bash
vertex_stack components:
├── artifact_store: gcs_store (gs://book-sales-deployment-artifacts)
├── orchestrator: vertex_orchestrator (Vertex AI)  
└── experiment_tracker: [MISSING] ← Only thing needed!
```

### What This Means
- ✅ **Persistent Storage**: GCS bucket already configured and working
- ✅ **Cloud Native**: Vertex AI orchestrator ready for deployment  
- ✅ **Cross-Environment**: Same storage accessible from local + cloud
- ⚠️ **Missing Piece**: Just need to add MLflow experiment tracker

---

## Issue Description

The MLflow experiment tracking is currently configured with a **local file path** on your `mlflow_stack`, but you have a much better solution available with your `vertex_stack`.

### Current Configuration
```bash
# ZenML MLflow tracker configuration
tracking_uri: file:///Users/zarif/Documents/Projects/book_sales_deployment/mlruns
```

### Stack Comparison

| Stack | Environment | MLflow Status | Solution Needed |
|-------|------------|---------------|-----------------|
| `mlflow_stack` | ✅ Local Development | **WORKS** | Local file access |
| `mlflow_stack` | ❌ Vertex AI | **FAILS** | No local file access |
| `vertex_stack` | ✅ Local Development | **MISSING** | Add MLflow tracker |
| `vertex_stack` | ✅ Vertex AI | **READY** | Add MLflow tracker |

**Key Insight**: vertex_stack just needs MLflow tracker added - no complex cloud setup required!

---

## Expected Deployment Errors

When deploying to cloud, you'll see errors like:

```bash
# Vertex AI Error
FileNotFoundError: [Errno 2] No such file or directory: '/Users/zarif/Documents/Projects/...'

# MLflow Error  
mlflow.exceptions.MlflowException: Unable to connect to tracking URI

# ZenML Error
Exception in step train_individual_arima_models_step: MLflow tracking failed
```

---

## Prerequisites - Steps 1 & 2

**⚠️ IMPORTANT**: Complete these steps before implementing any solution options below.

### Step 1: Create Cloud-Native MLflow Tracker

```bash
# Register new experiment tracker using your existing GCS bucket
zenml experiment-tracker register vertex_mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri="gs://book-sales-deployment-artifacts/mlflow-tracking"
```

**Expected Output**:
```
Successfully registered experiment tracker `vertex_mlflow_tracker`.
```

### Step 2: Add MLflow Tracker to vertex_stack

```bash
# Update your vertex_stack to include MLflow tracking
zenml stack update vertex_stack --experiment-tracker=vertex_mlflow_tracker
```

**Expected Output**:
```
Successfully updated stack `vertex_stack`.
```

### Verification of Prerequisites

```bash
# Verify the tracker was created
zenml experiment-tracker describe vertex_mlflow_tracker

# Verify vertex_stack now includes MLflow
zenml stack describe vertex_stack
# Should show: experiment_tracker: vertex_mlflow_tracker

# Check updated stack list
zenml stack list
```

**Status**: ⏸️ **PREREQUISITES PENDING** - Complete Steps 1 & 2 before proceeding with solutions

---

## Solution Options

### 🎯 Option 1: Use vertex_stack with MLflow (RECOMMENDED - Simplest)

**Prerequisites**: ✅ Complete Steps 1 & 2 above first

**Implementation**:

1. **Switch to vertex_stack**
   ```bash
   # Switch from mlflow_stack to vertex_stack for deployment
   zenml stack set vertex_stack
   ```

2. **Test the Setup**
   ```bash
   # Run pipeline with vertex_stack
   python3 pipelines/zenml_pipeline.py
   ```

**Pros**: 
- ✅ Uses your existing GCS bucket (`gs://book-sales-deployment-artifacts`)
- ✅ No additional cloud setup required
- ✅ Works locally AND on Vertex AI immediately
- ✅ Full MLflow functionality with persistent storage

**Cons**: 
- None! This is the optimal solution for your setup

---

### 🔧 Option 2: Update Existing mlflow_tracker (Alternative)

**Modify Current Tracker to Use GCS**

1. **Update Existing Tracker**
   ```bash
   # Point your current MLflow tracker to GCS
   zenml experiment-tracker update mlflow_tracker \
     --tracking_uri="gs://book-sales-deployment-artifacts/mlflow-tracking"
   ```

2. **Keep Using mlflow_stack**
   ```bash
   # Continue using your current stack
   zenml stack set mlflow_stack
   ```

**Pros**: 
- Keep using existing stack
- Still uses your GCS infrastructure

**Cons**: 
- Doesn't use Vertex AI orchestrator
- Less optimal than Option 1

---

### 🔄 Option 3: Dual Stack Approach (Advanced)

**Use Different Stacks for Different Purposes**

1. **Development Stack** (mlflow_stack)
   ```bash
   # For local development with immediate MLflow UI
   zenml stack set mlflow_stack
   python3 pipelines/zenml_pipeline.py  # Local development
   ```

2. **Production Stack** (vertex_stack)  
   ```bash
   # For cloud deployment
   zenml stack set vertex_stack
   python3 pipelines/zenml_pipeline.py  # Deploys to Vertex AI
   ```

3. **Switch Based on Need**
   ```bash
   # Quick switching between stacks
   alias dev-stack="zenml stack set mlflow_stack"
   alias prod-stack="zenml stack set vertex_stack"
   ```

**Pros**: 
- Best of both worlds
- Local development + cloud deployment
- Easy switching between environments

**Cons**: 
- Requires managing two stacks
- Need to remember to switch stacks

---

## Quick Implementation Guide

**After completing Prerequisites (Steps 1 & 2 above)**:

```bash
# Step 3: Switch to vertex_stack
zenml stack set vertex_stack

# Step 4: Test locally first
python3 pipelines/zenml_pipeline.py

# Step 5: Deploy to Vertex AI (will work identically)
# Your pipeline is now deployment-ready!
```

**Why This Works**:
- ✅ Uses your existing `gs://book-sales-deployment-artifacts` bucket
- ✅ MLflow data persists in cloud storage
- ✅ Same tracking works locally and on Vertex AI
- ✅ No additional cloud setup needed

---

## Testing Your Fix

### Local Testing with vertex_stack
```bash
# Switch to vertex_stack
zenml stack set vertex_stack

# Test pipeline creates MLflow runs in GCS
python3 pipelines/zenml_pipeline.py

# Check experiments were created
python3 check_all_experiments.py

# Verify GCS storage
gsutil ls gs://book-sales-deployment-artifacts/mlflow-tracking/
```

### Cloud Testing (Vertex AI Deployment)
```bash
# Deploy pipeline to Vertex AI
zenml stack set vertex_stack
python3 pipelines/zenml_pipeline.py  # Runs on Vertex AI

# Check results (same as local)
python3 check_all_experiments.py

# Verify artifacts in GCS
gsutil ls -r gs://book-sales-deployment-artifacts/
```

### Stack Switching Testing
```bash
# Test switching between stacks
zenml stack set mlflow_stack    # Local development
zenml stack set vertex_stack    # Cloud deployment

# Both should work with their respective tracking
```

---

## Troubleshooting

### Issue: "No MLflow runs visible after cloud deployment"
**Solution**: Check ZenML artifact store - experiments may be stored there instead of MLflow format

### Issue: "MLflow UI shows no experiments in cloud"
**Solution**: You need Option 1 (Cloud-Native MLflow) for full UI functionality

### Issue: "Permission denied accessing GCS bucket"
**Solution**: Ensure Vertex AI service account has Storage Admin permissions on MLflow bucket

### Issue: "Pipeline fails with tracking URI error"
**Solution**: Completely remove tracking_uri configuration and let ZenML handle it

---

## Current Status

**Development Environment**: ✅ Working (`mlflow_stack` with local tracking)  
**Prerequisites Status**: ⏸️ **PENDING** - Steps 1 & 2 need to be completed
**vertex_stack Ready**: ⚠️ **Waiting for Prerequisites** 
**Production Readiness**: ⏸️ **BLOCKED** - Complete prerequisites first

**Next Action**: Complete **Prerequisites Steps 1 & 2** before implementing any solution options.

---

## Related Files

- `pipelines/zenml_pipeline.py` - Contains MLflow logging code
- `check_all_experiments.py` - Script to verify MLflow experiments
- `zenml_mlflow_minimal.py` - Minimal test for ZenML MLflow integration

---

## Commands Reference

### vertex_stack Setup (Recommended)
```bash
# Create cloud MLflow tracker for vertex_stack
zenml experiment-tracker register vertex_mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri="gs://book-sales-deployment-artifacts/mlflow-tracking"

# Add to vertex_stack
zenml stack update vertex_stack --experiment-tracker=vertex_mlflow_tracker

# Switch stacks
zenml stack set vertex_stack  # Production
zenml stack set mlflow_stack  # Local development

# Check stack status
zenml stack list
zenml stack describe vertex_stack
```

### Verification Commands
```bash
# Test MLflow connection
python3 -c "import mlflow; print('URI:', mlflow.get_tracking_uri())"

# List all experiments
python3 check_all_experiments.py

# Check GCS storage
gsutil ls gs://book-sales-deployment-artifacts/mlflow-tracking/
```

---

**✅ SUMMARY**: Your vertex_stack infrastructure is already deployment-ready - just add the MLflow tracker!