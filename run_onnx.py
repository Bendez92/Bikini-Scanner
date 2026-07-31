from bikini_scanner.config import ScannerConfig
from bikini_scanner.run import main_onnx


def main_with_onnx_config(config: ScannerConfig) -> int:
    return main_onnx(config_override=config)


if __name__ == "__main__":
    raise SystemExit(main_onnx())
