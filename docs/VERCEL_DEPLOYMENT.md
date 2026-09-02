# Vercel Deployment

## Project settings

- GitHub repository: `https://github.com/SunidhiDeekonda/phishing-ML-URL-detector`
- Production branch: `main`
- Root directory: `./`
- Framework: FastAPI
- FastAPI entry point: `app.main:app`
- Python runtime: `3.12`
- Build command: empty
- Output directory: empty

## Import and deploy

1. Sign in to Vercel using Sunidhi's account.
2. Open **Add New -> Project**.
3. Import `SunidhiDeekonda/phishing-ML-URL-detector` from GitHub.
4. Keep the root directory at the repository root.
5. Leave build-command and output-directory overrides empty.
6. Do not add environment variables; the application requires no secrets or external services.
7. Select **Deploy**.

Vercel detects `app/main.py` as FastAPI and installs `requirements.txt`. Deployment packaging retains `app/`, `src/`, `models/char_cnn.onnx`, `models/lightgbm.onnx`, `models/lightgbm_calibration.json`, and `data/processed/features.csv`. Training data, reports, tests, pickle model artifacts, and local files are excluded from the function bundle but remain available in GitHub.

## Environment variables

No user-defined environment variables are required. The application has no API key, database URL, authentication secret, or remote model endpoint. Do not upload `.env` files, GitHub tokens, or Vercel tokens.

Vercel supplies system environment variables automatically. They do not need to be copied into the repository.

## Logs

- Build logs: open the Vercel project, choose **Deployments**, select a deployment, and open **Build Logs**.
- Runtime logs: open the Vercel project and select **Logs**.
- Continuous-integration logs: open GitHub and select **Actions -> Reproducibility CI**.

Build and runtime logs are generated only after a deployment exists. They are operational records and should not be committed to Git.

## Post-deployment checks

Open the deployment root to load the demonstration interface. Then open:

```text
https://<project-domain>/health
```

A healthy response reports `lightgbm_loaded` and `cnn_loaded` as `true`, with `device` set to `cpu`. Predictions inspect URL text only; they do not perform DNS requests or visit submitted domains.

If a deployment fails, retain the complete Vercel build log and address the first reported error. Do not retrain either model or replace audited artifacts as part of deployment troubleshooting.
