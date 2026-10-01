# Contributing to Phishing ML URL Detector

Thank you for your interest in contributing! This guide will help you get
started.

## Local Setup

```bash
git clone https://github.com/SunidhiDeekonda/phishing-ML-URL-detector.git
cd phishing-ML-URL-detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running Tests

The project uses **pytest**. After activating your virtual environment:

```bash
pytest
```

All tests should pass before you open a pull request.

## Running the Local Demo

```bash
uvicorn app.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Guidelines

- **Do not modify trained model files** in `models/`. These are production
  artifacts used by the deployed application.
- **Do not change prediction logic** in `app/inference.py` or ensemble weights
  without discussion.
- **Do not alter deployment configuration** (e.g., `vercel.json`,
  `requirements.txt`) unless you have confirmed the change with a maintainer.
- Keep pull requests small and focused on a single improvement.
- Write clear commit messages that describe *what* changed and *why*.

## What Makes a Good Contribution?

- Adding or improving docstrings and code comments.
- Writing new tests for existing functions.
- Improving documentation (README, guides, inline comments).
- Fixing typos or clarifying developer instructions.
- Reporting bugs via GitHub Issues with clear reproduction steps.

## Code of Conduct

Be respectful and constructive in all interactions. This is an academic project
and contributions of all sizes are welcome.
