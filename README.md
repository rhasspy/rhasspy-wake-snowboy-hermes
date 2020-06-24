# Rhasspy Wake Snowboy Hermes

[![Continous Integration](https://github.com/rhasspy/rhasspy-wake-snowboy-hermes/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-wake-snowboy-hermes/actions)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-wake-snowboy-hermes.svg)](https://github.com/rhasspy/rhasspy-wake-snowboy-hermes/blob/master/LICENSE)

Implements `hermes/hotword` functionality from [Hermes protocol](https://docs.snips.ai/reference/hermes) using [snowboy](https://snowboy.kitt.ai).

## Requirements

* Python 3.7
* [snowboy](https://snowboy.kitt.ai)

## Installation

```bash
$ git clone https://github.com/rhasspy/rhasspy-wake-snowboy-hermes
$ cd rhasspy-wake-snowboy-hermes
$ ./configure
$ make
$ make install
```

## Running

```bash
$ bin/rhasspy-wake-snowboy-hermes <ARGS>
```

## Command-Line Options

```
usage: rhasspy-wake-snowboy-hermes [-h] --model MODEL [MODEL ...]
                                   [--model-dir MODEL_DIR]
                                   [--wakeword-id WAKEWORD_ID] [--stdin-audio]
                                   [--udp-audio UDP_AUDIO UDP_AUDIO UDP_AUDIO]
                                   [--host HOST] [--port PORT]
                                   [--username USERNAME] [--password PASSWORD]
                                   [--tls] [--tls-ca-certs TLS_CA_CERTS]
                                   [--tls-certfile TLS_CERTFILE]
                                   [--tls-keyfile TLS_KEYFILE]
                                   [--tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}]
                                   [--tls-version TLS_VERSION]
                                   [--tls-ciphers TLS_CIPHERS]
                                   [--site-id SITE_ID] [--debug]
                                   [--log-format LOG_FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL [MODEL ...]
                        Snowboy model settings (model, sensitivity,
                        audio_gain, apply_frontend)
  --model-dir MODEL_DIR
                        Directories with snowboy models
  --wakeword-id WAKEWORD_ID
                        Wakeword IDs of each keyword (default: use file name)
  --stdin-audio         Read WAV audio from stdin
  --udp-audio UDP_AUDIO UDP_AUDIO UDP_AUDIO
                        Host/port/siteId for UDP audio input
  --host HOST           MQTT host (default: localhost)
  --port PORT           MQTT port (default: 1883)
  --username USERNAME   MQTT username
  --password PASSWORD   MQTT password
  --tls                 Enable MQTT TLS
  --tls-ca-certs TLS_CA_CERTS
                        MQTT TLS Certificate Authority certificate files
  --tls-certfile TLS_CERTFILE
                        MQTT TLS certificate file (PEM)
  --tls-keyfile TLS_KEYFILE
                        MQTT TLS key file (PEM)
  --tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}
                        MQTT TLS certificate requirements (default:
                        CERT_REQUIRED)
  --tls-version TLS_VERSION
                        MQTT TLS version (default: highest)
  --tls-ciphers TLS_CIPHERS
                        MQTT TLS ciphers to use
  --site-id SITE_ID     Hermes site id(s) to listen for (default: all)
  --debug               Print DEBUG messages to the console
  --log-format LOG_FORMAT
                        Python logger format
```
