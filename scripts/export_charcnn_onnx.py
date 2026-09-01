"""Export the saved Char-CNN checkpoint to the lightweight deployment format.

Training uses PyTorch, but the FastAPI application uses the generated ONNX file
at runtime so it can run within Vercel's function bundle limit.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.train_charcnn import CharCNN, MAX_SEQUENCE_LENGTH


CHECKPOINT_PATH = Path("models/char_cnn.pt")
OUTPUT_PATH = Path("models/char_cnn.onnx")


def main() -> None:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = CharCNN()
    model.load_state_dict(state_dict)
    model.eval()

    example_input = torch.zeros((1, MAX_SEQUENCE_LENGTH), dtype=torch.long)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example_input,
            OUTPUT_PATH,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )

    print(f"Exported {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
