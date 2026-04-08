""" 
DEPLOYMENT GUIDE: DataCleanerEnv → Hugging Face Spaces

This document provides step-by-step instructions for deploying DataCleanerEnv to HF Spaces.
"""

## STEP 1: Prepare Your GitHub Repository

1. Initialize git if not already done:
   ```bash
   cd e:\hackathon
   git init
   git add .
   git commit -m "Initial commit: DataCleanerEnv OpenEnv environment"
   ```

2. Push to GitHub:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git branch -M main
   git push -u origin main
   ```

---

## STEP 2: Create Hugging Face Space

1. Go to: https://huggingface.co/spaces

2. Click "Create new Space"

3. Fill in:
   - **Name**: `dataclean-env` (or similar)
   - **Owner**: Your HF username
   - **License**: OpenRAIL (or your choice)
   - **SDK**: Docker
   - **Repository name**: `dataclean-env`

4. Click "Create Space"

---

## STEP 3: Set Environment Variables

1. In your HF Space settings (⚙️ Settings):
   
   Add secrets:
   ```
   OPENAI_API_KEY = sk-...your-key...
   MODEL_NAME = gpt-4-turbo
   API_BASE_URL = https://api.openai.com/v1
   ```

2. Save settings

---

## STEP 4: Connect Git Repository & Deploy

Option A: **Using HF Git URL (Recommended)**

```bash
cd e:\hackathon

# Add HF remote
git remote add hf https://huggingface.co/spaces/<username>/<space-name>.git

# Push to HF (triggers auto-build)
git push hf main
```

Option B: **Direct File Upload**
1. Go to Space page
2. Click "Files and versions"
3. Upload all files via web interface

---

## STEP 5: Wait for Build

HF will:
1. Download your Dockerfile
2. Install dependencies from requirements.txt
3. Build Docker image
4. Start container on port 8080

**Check status**: Space page shows build progress in top-right

**Typical build time**: 3-5 minutes

---

## STEP 6: Verify Space Works

Once deployed:

1. **Health check**: Space page shows "Running" status

2. **API endpoints available**:
   - `POST /reset` → Initialize environment
   - `POST /step` → Execute action
   - `GET /state` → Get internal state

3. **Test baseline**: Run inference script against Space:
   ```bash
   export HF_SPACE_API="https://<username>-<space-name>.hf.space"
   python inference.py --use-hf-api
   ```

---

## STEP 7: Pre-Submission Final Checklist

Before submitting to hackathon:

### Automated Validation
```bash
# 1. OpenEnv spec compliance
pip install openenv-core
openenv validate

# 2. Docker build locally
docker build -t dataclean-env:latest .

# 3. Test baseline (< 20 min)
export OPENAI_API_KEY="sk-..."
time python inference.py
```

### Manual Verification
- [ ] HF Space URL responds (status = Running)
- [ ] Can call `/reset` endpoint
- [ ] Can call `/step` endpoint with valid action
- [ ] Baseline inference script completes without errors
- [ ] All 3 tasks produce different scores (not uniform)
- [ ] Scores are in [0.0, 1.0] range
- [ ] README documents everything

---

## STEP 8: Monitor Space

Once live:
- HF Spaces dashboard shows resource usage
- Logs visible in "Runtime Logs" tab
- Can restart container from Space settings if needed

---

## TROUBLESHOOTING

### Space won't build
- Check Dockerfile syntax: `docker build .` locally first
- Ensure all dependencies in requirements.txt
- Check file paths are correct (case-sensitive on Linux)

### Inference script timeout
- HF limit: 20 minutes per request
- If baseline needs > 20 min:
  - Reduce MAX_STEPS_PER_TASK in inference.py
  - Optimize reward calculation
  - Use faster model (e.g., gpt-4-turbo-preview)

### API key not working
- Verify OPENAI_API_KEY is valid
- Set in Space secrets (⚙️ Settings → Secrets)
- Restart container after updating secrets

### Task datasets not found
- Ensure datasets/ directory has task_*.json files
- Check file names: task_easy.json, task_medium.json, task_hard.json
- Run: `python -m environment.data_loader` locally to regenerate

---

## SUBMISSION

When ready to submit:

1. **Copy Space URL**: `https://<username>-<space-name>.hf.space`

2. **Verify checklist**:
   - ✓ Space is public
   - ✓ Responds to `/reset` (200 status)
   - ✓ openenv validate passes
   - ✓ Dockerfile builds
   - ✓ Baseline runs in < 20 min
   - ✓ All 3 tasks with graders
   - ✓ README complete

3. **Submit to hackathon**:
   - Space URL
   - GitHub repo URL
   - Brief summary

---

## ADDITIONAL RESOURCES

- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces-overview
- **HF Spaces Docker**: https://huggingface.co/docs/hub/spaces-docker
- **OpenEnv Spec**: https://github.com/openenv/openenv-spec
- **Submission Form**: [hackathon link]

---

**Estimated time to deployment**: 30 minutes (after code is ready)

Good luck with submission! 🚀
