"""CLI entrypoint for mandible-clarity."""
from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mandible_clarity.engine import ClarityEngine, detect_hardware
else:
    from .engine import ClarityEngine, detect_hardware


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mandible-clarity",
        description="Headless DeepFilterNet studio AI vocal de-noiser and dereverberation sidecar for Mandible.",
    )
    sub = parser.add_subparsers(dest="command")

    # Command: info
    sub.add_parser("info", help="Inspect hardware acceleration and inference backend status")

    # Command: enhance
    enh_parser = sub.add_parser("enhance", help="Enhance an audio file (remove noise and room reverb)")
    enh_parser.add_argument("--input", "-i", required=True, help="Input audio file (.wav, .flac, .mp3)")
    enh_parser.add_argument("--output", "-o", required=True, help="Output destination path")
    enh_parser.add_argument("--atten-limit", "-a", type=float, default=35.0, help="Max attenuation limit in dB (default: 35.0)")
    enh_parser.add_argument("--dereverb", action="store_true", default=True, help="Enable acoustic dereverberation (default: True)")
    enh_parser.add_argument("--no-dereverb", dest="dereverb", action="store_false", help="Disable acoustic dereverberation")
    enh_parser.add_argument("--model", "-m", default=None, help="Optional custom ONNX model path")

    args = parser.parse_args()

    if args.command == "info":
        hw = detect_hardware()
        data = {
            "available": hw.available,
            "device": hw.device_name,
            "backend": hw.backend,
            "supports_cuda": hw.supports_cuda,
            "supports_directml": hw.supports_directml,
        }
        print(json.dumps(data, indent=2))
        return 0

    if args.command == "enhance":
        if not os.path.isfile(args.input):
            print(f"Error: input file '{args.input}' not found.", file=sys.stderr)
            return 1

        engine = ClarityEngine(
            model_path=args.model,
            atten_limit_db=args.atten_limit,
            dereverb=args.dereverb,
        )

        def _prog(p: float):
            pct = int(p * 100)
            sys.stdout.write(f"\rEnhancing audio: {pct}% complete...")
            sys.stdout.flush()

        print(f"Processing '{args.input}' with Mandible Clarity...")
        engine.enhance_file(
            input_path=args.input,
            output_path=args.output,
            atten_limit_db=args.atten_limit,
            dereverb=args.dereverb,
            progress_cb=_prog,
        )
        print(f"\nSaved enhanced vocal audio to '{args.output}'.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
