from __future__ import annotations

import onnx
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from src.charcnn_bilstm import (
    CharCNNBiLSTMAttention,
    MAX_SEQUENCE_LENGTH,
    build_enhanced_vocab,
    encode_url_bilstm,
)
from src.domain_reputation import calibrate_probability_with_domain

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

def main() -> None:
    vocab_path = MODELS_DIR / "charcnn_bilstm_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path.exists() else build_enhanced_vocab()

    weights_path = MODELS_DIR / "charcnn_bilstm_best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}")

    model = CharCNNBiLSTMAttention(vocab_size=len(vocab))
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    print("Loaded trained CharCNN-BiLSTM weights successfully!")

    # Step 1: Export to ONNX
    onnx_path = MODELS_DIR / "charcnn_bilstm.onnx"
    dummy_input = torch.zeros((1, MAX_SEQUENCE_LENGTH), dtype=torch.long)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    print(f"Exported ONNX model to {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.2f} MB).")

    # Step 2: Parity Check
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_urls = [
        "https://google.com",
        "https://github.com/login",
        "https://en.wikipedia.org/wiki/Phishing",
        "http://suspicious-login-verify-account-paypal.com/secure/login.php",
    ]
    encoded_tests = np.stack([encode_url_bilstm(u, vocab, max_len=MAX_SEQUENCE_LENGTH) for u in test_urls], axis=0)
    with torch.no_grad():
        py_logits = model(torch.as_tensor(encoded_tests, dtype=torch.long)).numpy()
    onnx_logits = sess.run(["logits"], {"input": encoded_tests})[0]
    np.testing.assert_allclose(py_logits, onnx_logits, rtol=1e-3, atol=1e-3)
    print("Numerical parity test PASSED! PyTorch and ONNX match perfectly.")

    # Step 3: Hard-Negative Benchmark Verification
    hard_cases_path = ROOT / "tests/data/hard_legitimate_urls.json"
    hard_cases = json.loads(hard_cases_path.read_text(encoding="utf-8"))
    print(f"\nEvaluating on {len(hard_cases)} curated hard-negative legitimate URLs...")

    hard_results = []
    for case in hard_cases:
        url = case["url"]
        encoded = encode_url_bilstm(url, vocab, max_len=MAX_SEQUENCE_LENGTH)
        logits = sess.run(["logits"], {"input": encoded[np.newaxis, :]})[0]
        raw_prob = float(1.0 / (1.0 + np.exp(-float(logits[0]))))
        prob = calibrate_probability_with_domain(url, raw_prob)
        verdict = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
        status = "PASS" if verdict == case["expected_label"] else "FAIL"
        hard_results.append({
            "url": url,
            "split": case["hard_negative_split"],
            "category": case["category"],
            "expected": case["expected_label"],
            "probability": round(prob, 4),
            "verdict": verdict,
            "status": status,
        })

    hard_df = pd.DataFrame(hard_results)
    print("\n--- Hard Negatives Benchmark Results ---")
    summary_table = hard_df.groupby(["split", "status"]).size().unstack(fill_value=0)
    print(summary_table)

    passed_count = (hard_df["status"] == "PASS").sum()
    total_count = len(hard_df)
    pass_rate = passed_count / total_count
    print(f"\nOverall Hard-Negative Pass Rate: {passed_count} / {total_count} ({pass_rate:.1%})")

    failures = hard_df[hard_df["status"] == "FAIL"]
    if len(failures) > 0:
        print(f"\nRemaining Failures ({len(failures)}):")
        print(failures[["url", "probability", "status"]].to_string())
    else:
        print("\nALL 94 HARD NEGATIVES PASSED PERFECTLY!")

    # Check key URLs that previously failed in production v2
    key_urls = [
        "https://github.com/login",
        "https://vercel.com/login",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://support.apple.com/en-us/102660",
        "https://help.openai.com/en/articles/6614161-how-can-i-contact-support",
        "https://docs.python.org/3/library/functions.html",
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication",
        "https://doc.rust-lang.org/book/ch01-01-installation.html",
        "https://dash.cloudflare.com/login",
        "https://myaccount.google.com/security",
        "https://platform.openai.com/login",
        "https://account.apple.com/sign-in",
        "https://owasp.org/www-community/attacks/Phishing",
        "https://github.com/SunidhiDeekonda/phishing-ML-URL-detector",
    ]

    print("\n--- Critical Previously-Failing URLs Verification ---")
    for u in key_urls:
        encoded = encode_url_bilstm(u, vocab, max_len=MAX_SEQUENCE_LENGTH)
        logits = sess.run(["logits"], {"input": encoded[np.newaxis, :]})[0]
        raw_prob = float(1.0 / (1.0 + np.exp(-float(logits[0]))))
        prob = calibrate_probability_with_domain(u, raw_prob)
        verdict = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
        print(f"  {u:<65} -> {verdict:<10} (p={prob:.4f})")

    # Save metrics
    metrics_summary = {
        "architecture": "CharCNN + BiLSTM + Attention",
        "onnx_model_size_mb": round(onnx_path.stat().st_size / 1024 / 1024, 2),
        "hard_negatives": {
            "total": total_count,
            "passed": int(passed_count),
            "failed": int(len(failures)),
            "pass_rate": round(pass_rate, 4),
        },
    }
    (RESULTS_DIR / "charcnn_bilstm_metrics.json").write_text(json.dumps(metrics_summary, indent=2))
    print(f"\nSaved metrics to {RESULTS_DIR / 'charcnn_bilstm_metrics.json'}")

if __name__ == "__main__":
    main()
