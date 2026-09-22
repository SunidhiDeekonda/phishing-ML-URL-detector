# Pre-extension Repository Audit

Audit date: 2026-09-22

- Branch: `main`
- Remote: `https://github.com/SunidhiDeekonda/phishing-ML-URL-detector.git`
- Existing changes: none; working tree clean and synchronized.
- Preserved work: all preprocessing, model, ONNX portability, deployment, documentation, and root-URL canonicalization commits.
- Modified/untracked files before extension: none.
- Model artifacts: original LightGBM pickle/ONNX and character-CNN PyTorch/ONNX artifacts plus calibration metadata.
- Tests before extension: 27 passed, 0 failed (`pytest -q`, 7.17 seconds).
- Deployment: Vercel configuration, FastAPI entry point, static frontend, and ONNX runtime artifacts present.
- README/report: completed reproduction, deployment documentation, and academic Markdown report present.
- Data: 20,000 balanced rows, 36 features, domain-separated 14,002/2,000/3,998 indices.
- Safety: generated URLs are inert strings only; DNS, HTTP, retrieval, and execution are forbidden.
