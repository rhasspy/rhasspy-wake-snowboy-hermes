"""Hermes MQTT service for Rhasspy wakeword with snowboy"""
import argparse
import asyncio
import dataclasses
import itertools
import json
import logging
import os
import sys
import typing
from pathlib import Path

import paho.mqtt.client as mqtt

import rhasspyhermes.cli as hermes_cli

from . import SnowboyModel, WakeHermesMqtt

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("rhasspywake_snowboy_hermes")

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    parser = argparse.ArgumentParser(prog="rhasspy-wake-snowboy-hermes")
    parser.add_argument(
        "--model",
        required=True,
        action="append",
        nargs="+",
        help="Snowboy model settings (model, sensitivity, audio_gain, apply_frontend)",
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help="Directories with snowboy models",
    )
    parser.add_argument(
        "--wakeword-id",
        action="append",
        help="Wakeword IDs of each keyword (default: use file name)",
    )
    parser.add_argument(
        "--stdin-audio", action="store_true", help="Read WAV audio from stdin"
    )
    parser.add_argument(
        "--udp-audio",
        nargs=3,
        action="append",
        help="Host/port/siteId for UDP audio input",
    )

    hermes_cli.add_hermes_args(parser)
    args = parser.parse_args()

    hermes_cli.setup_logging(args)
    _LOGGER.debug(args)

    if args.model_dir:
        args.model_dir = [Path(d) for d in args.model_dir]

    # Use embedded models too
    args.model_dir.append(_DIR / "models")

    # Load model settings
    models: typing.List[SnowboyModel] = []

    for model_settings in args.model:
        model_path = Path(model_settings[0])

        if not model_path.is_file():
            # Resolve relative to model directories
            for model_dir in args.model_dir:
                maybe_path = model_dir / model_path.name
                if maybe_path.is_file():
                    model_path = maybe_path
                    break

        _LOGGER.debug("Loading model from %s", str(model_path))
        model = SnowboyModel(model_path=model_path)

        if len(model_settings) > 1:
            model.sensitivity = model_settings[1]

        if len(model_settings) > 2:
            model.audio_gain = float(model_settings[2])

        if len(model_settings) > 3:
            model.apply_frontend = model_settings[3].strip().lower() == "true"

        models.append(model)

    wakeword_ids = [
        kn[1]
        for kn in itertools.zip_longest(
            args.model, args.wakeword_id or [], fillvalue=""
        )
    ]

    if args.stdin_audio:
        # Read WAV from stdin, detect, and exit
        client = None
        hermes = WakeHermesMqtt(client, models, wakeword_ids)

        hermes.load_detectors()

        if os.isatty(sys.stdin.fileno()):
            print("Reading WAV data from stdin...", file=sys.stderr)

        wav_bytes = sys.stdin.buffer.read()

        # Print results as JSON
        for result in hermes.handle_audio_frame(wav_bytes):
            result_dict = dataclasses.asdict(result)
            json.dump(result_dict, sys.stdout, ensure_ascii=False)

        return

    udp_audio = []
    if args.udp_audio:
        udp_audio = [
            (host, int(port), site_id) for host, port, site_id in args.udp_audio
        ]

    # Listen for messages
    client = mqtt.Client()
    hermes = WakeHermesMqtt(
        client,
        models,
        wakeword_ids,
        model_dirs=args.model_dir,
        udp_audio=udp_audio,
        site_ids=args.site_id,
    )

    hermes.load_detectors()

    _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
    hermes_cli.connect(client, args)

    client.loop_start()

    try:
        # Run event loop
        asyncio.run(hermes.handle_messages_async())
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")
        client.loop_stop()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
