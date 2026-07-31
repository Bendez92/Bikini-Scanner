from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from bikini_scanner.config import DEFAULT_MODEL_NAME

VISION_ONNX_NAME = "clip_vision.onnx"
TEXT_ONNX_NAME = "clip_text.onnx"


class VisionWrapper(torch.nn.Module):
    def __init__(self, model: CLIPModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        return self.model.visual_projection(vision_outputs.pooler_output)


class TextWrapper(torch.nn.Module):
    def __init__(self, model: CLIPModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.model.text_projection(text_outputs.pooler_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Bikini Scanner CLIP towers to ONNX")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default="models")
    return parser


def export_onnx(model_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    vision_wrapper = VisionWrapper(model)
    text_wrapper = TextWrapper(model)
    vision_wrapper.eval()
    text_wrapper.eval()
    vision_path = output_dir / VISION_ONNX_NAME
    text_path = output_dir / TEXT_ONNX_NAME
    pixel_values = torch.randn(1, 3, 224, 224)
    text_inputs = processor(
        text=["a person in a bikini"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
    )
    torch.onnx.export(
        vision_wrapper,
        pixel_values,
        vision_path,
        input_names=["pixel_values"],
        output_names=["projected_features"],
        dynamic_axes={"pixel_values": {0: "batch"}, "projected_features": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    torch.onnx.export(
        text_wrapper,
        (text_inputs["input_ids"], text_inputs["attention_mask"]),
        text_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["projected_features"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "projected_features": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )
    print(f"Exported {vision_path}")
    print(f"Exported {text_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_onnx(args.model_name, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
