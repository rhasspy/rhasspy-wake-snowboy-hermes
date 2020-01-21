# Rhasspy Wake Snowboy Hermes

[![Continous Integration](https://github.com/rhasspy/rhasspy-wake-snowboy-hermes/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-wake-snowboy-hermes/actions)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-wake-snowboy-hermes.svg)](https://github.com/rhasspy/rhasspy-wake-snowboy-hermes/blob/master/LICENSE)

Implements `hermes/hotword` functionality from [Hermes protocol](https://docs.snips.ai/reference/hermes) using [snowboy](https://snowboy.kitt.ai).

## Running With Docker

```bash
docker run -it rhasspy/rhasspy-wake-snowboy-hermes:<VERSION> <ARGS>
```

## Building From Source

Clone the repository and create the virtual environment:

```bash
git clone https://github.com/rhasspy/rhasspy-wake-snowboy-hermes.git
cd rhasspy-wake-snowboy-hermes
make venv
```

Run the `bin/rhasspy-wake-snowboy-hermes` script to access the command-line interface:

```bash
bin/rhasspy-wake-snowboy-hermes --help
```

## Building the Debian Package

Follow the instructions to build from source, then run:

```bash
source .venv/bin/activate
make debian
```

If successful, you'll find a `.deb` file in the `dist` directory that can be installed with `apt`.

## Building the Docker Image

Follow the instructions to build from source, then run:

```bash
source .venv/bin/activate
make docker
```

This will create a Docker image tagged `rhasspy/rhasspy-wake-snowboy-hermes:<VERSION>` where `VERSION` comes from the file of the same name in the source root directory.

NOTE: If you add things to the Docker image, make sure to whitelist them in `.dockerignore`.

## Command-Line Options

```
usage: rhasspywake_snowboy_hermes [-h] --model MODEL [MODEL ...]
                                  [--wakewordId WAKEWORDID] [--stdin-audio]
                                  [--host HOST] [--port PORT]
                                  [--siteId SITEID] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL [MODEL ...]
                        Snowboy model settings (model, sensitivity,
                        audio_gain, apply_frontend)
  --wakewordId WAKEWORDID
                        Wakeword IDs of each keyword (default: default)
  --stdin-audio         Read WAV audio from stdin
  --host HOST           MQTT host (default: localhost)
  --port PORT           MQTT port (default: 1883)
  --siteId SITEID       Hermes siteId(s) to listen for (default: all)
  --debug               Print DEBUG messages to the console
```
