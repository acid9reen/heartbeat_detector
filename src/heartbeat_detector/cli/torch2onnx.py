import argparse
from pathlib import Path

import torch
from heartbeat_detector.models.unet_1d import UNet1d


class Torch2OnnxNamespace(argparse.Namespace):
    model_path: Path
    out_filename: str | None
    input_size: int


def parse_args() -> Torch2OnnxNamespace:
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
        help='Filename for output.onnx model',
        default=None,
    )

    parser.add_argument(
        '-i', '--input_size',
        type=int,
        help='Input size for given model',
        default=10_000,
    )

    return parser.parse_args(namespace=Torch2OnnxNamespace())


def main() -> int:
    args = parse_args()

    output_filename = (
        args.model_path.stem + '.onnx' if args.out_filename is None
        else args.out_filename
    )

    model = torch.load(args.model_path).to('cpu')
    model.eval()

    dummy_input = torch.randn(1, 1, args.input_size, device='cpu')

    torch.onnx.export(
        model,
        dummy_input,
        output_filename,
        export_params=True,
        opset_version=13,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
