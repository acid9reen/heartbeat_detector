import argparse
from pathlib import Path

import torch


class Torch2TorchscriptNamespace(argparse.Namespace):
    model_path: Path
    out_filename: str | None
    input_size: int


def parse_args() -> Torch2TorchscriptNamespace:
    parser = argparse.ArgumentParser(
        description='Convert .pth model to .onnx model',
    )

    parser.add_argument(
        'model_path',
        type=Path,
        help='Path to .pth model',
    )

    parser.add_argument(
        '-o', '--out_filename',
        help='Filename for output.pt model',
        default=None,
    )

    parser.add_argument(
        '-i', '--input_size',
        type=int,
        help='Input size for given model',
        default=10_000,
    )

    return parser.parse_args(namespace=Torch2TorchscriptNamespace())


def main() -> int:
    args = parse_args()

    output_filename = (
        args.model_path.stem + '.pt' if args.out_filename is None
        else args.out_filename
    )

    model = torch.load(args.model_path).to('cpu')
    model.eval()

    model_scripted = torch.jit.script(model)
    model_scripted.save(output_filename) # type: ignore

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
