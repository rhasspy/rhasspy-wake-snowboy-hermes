"""Hermes MQTT service for Rhasspy wakeword with snowboy"""
import argparse
import asyncio
import itertools
import json
import logging
import os
import sys
import typing
from pathlib import Path

import attr
import paho.mqtt.client as mqtt

from . import SnowboyModel, WakeHermesMqtt

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("rhasspywake_snowboy_hermes")

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    parser = argparse.ArgumentParser(prog="rhasspywake_snowboy_hermes")
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
        "--wakewordId",
        action="append",
        help="Wakeword IDs of each keyword (default: default)",
    )
    parser.add_argument(
        "--stdin-audio", action="store_true", help="Read WAV audio from stdin"
    )
    parser.add_argument(
        "--udp-audio-port", type=int, help="Also listen for WAV audio on UDP"
    )
    parser.add_argument(
        "--host", default="localhost", help="MQTT host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=1883, help="MQTT port (default: 1883)"
    )
    parser.add_argument(
        "--siteId",
        action="append",
        help="Hermes siteId(s) to listen for (default: all)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    parser.add_argument(
        "--log-format",
        default="[%(levelname)s:%(asctime)s] %(name)s: %(message)s",
        help="Python logger format",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=args.log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=args.log_format)

    _LOGGER.debug(args)

    try:
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
                args.model, args.wakewordId or [], fillvalue="default"
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
                result_dict = attr.asdict(result)
                json.dump(result_dict, sys.stdout)

            return

        loop = asyncio.get_event_loop()

        # Listen for messages
        client = mqtt.Client()
        hermes = WakeHermesMqtt(
            client,
            models,
            wakeword_ids,
            model_dirs=args.model_dir,
            udp_audio_port=args.udp_audio_port,
            siteIds=args.siteId,
            loop=loop,
        )

        hermes.load_detectors()

        _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
        client.connect(args.host, args.port)

        client.loop_start()

        # Run event loop
        hermes.loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
