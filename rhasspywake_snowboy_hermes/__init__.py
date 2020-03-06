"""Hermes MQTT server for Rhasspy wakeword with snowboy"""
import io
import json
import logging
import queue
import socket
import subprocess
import threading
import typing
import wave
from pathlib import Path

import attr
from rhasspyhermes.audioserver import AudioFrame
from rhasspyhermes.base import Message
from rhasspyhermes.wake import (
    GetHotwords,
    Hotword,
    HotwordDetected,
    HotwordError,
    Hotwords,
    HotwordToggleOff,
    HotwordToggleOn,
)

WAV_HEADER_BYTES = 44
_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


@attr.s(auto_attribs=True, slots=True)
class SnowboyModel:
    """Settings for a single snowboy model"""

    model_path: Path
    sensitivity: str = "0.5"
    audio_gain: float = 1.0
    apply_frontend: bool = False


# -----------------------------------------------------------------------------


class WakeHermesMqtt:
    """Hermes MQTT server for Rhasspy wakeword with snowboy."""

    def __init__(
        self,
        client,
        models: typing.List[SnowboyModel],
        wakeword_ids: typing.List[str],
        model_dirs: typing.Optional[typing.List[Path]] = None,
        siteIds: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        chunk_size: int = 960,
        udp_audio_port: typing.Optional[int] = None,
        udp_chunk_size: int = 2048,
    ):
        self.client = client
        self.models = models
        self.wakeword_ids = wakeword_ids
        self.model_dirs = model_dirs or []

        self.siteIds = siteIds or []
        self.enabled = enabled

        # Required audio format
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels

        self.chunk_size = chunk_size

        # Queue of WAV audio chunks to process (plus siteId)
        self.wav_queue: queue.Queue = queue.Queue()

        # Listen for raw audio on UDP too
        self.udp_audio_port = udp_audio_port
        self.udp_chunk_size = udp_chunk_size

        # siteId used for detections from UDP
        self.udp_siteId = "default" if not self.siteIds else self.siteIds[0]

        # Topics to listen for WAV chunks on
        self.audioframe_topics: typing.List[str] = []
        for siteId in self.siteIds:
            self.audioframe_topics.append(AudioFrame.topic(siteId=siteId))

        self.first_audio: bool = True

        self.audio_buffer = bytes()

        # Load detector
        self.detectors: typing.List[typing.Any] = []
        self.model_ids: typing.List[str] = []

    # -------------------------------------------------------------------------

    def load_detectors(self):
        """Load snowboy detectors from models"""
        from snowboy import snowboydecoder, snowboydetect

        self.model_ids = []
        self.detectors = []

        for model in self.models:
            assert model.model_path.is_file(), f"Missing {model.model_path}"
            _LOGGER.debug("Loading snowboy model: %s", model)

            detector = snowboydetect.SnowboyDetect(
                snowboydecoder.RESOURCE_FILE.encode(), str(model.model_path).encode()
            )

            detector.SetSensitivity(model.sensitivity.encode())
            detector.SetAudioGain(model.audio_gain)
            detector.ApplyFrontend(model.apply_frontend)

            self.detectors.append(detector)
            self.model_ids.append(model.model_path.stem)

    # -------------------------------------------------------------------------

    def handle_audio_frame(self, wav_bytes: bytes, siteId: str = "default"):
        """Process a single audio frame"""
        self.wav_queue.put((wav_bytes, siteId))

    def handle_detection(
        self, model_index, siteId="default"
    ) -> typing.Union[HotwordDetected, HotwordError]:
        """Handle a successful hotword detection"""
        try:
            assert len(self.model_ids) > model_index, f"Missing {model_index} in models"

            return HotwordDetected(
                siteId=siteId,
                modelId=self.model_ids[model_index],
                currentSensitivity=self.models[model_index].sensitivity,
                modelVersion="",
                modelType="personal",
            )
        except Exception as e:
            _LOGGER.exception("handle_detection")
            return HotwordError(error=str(e), context=str(model_index), siteId=siteId)

    def handle_get_hotwords(
        self, get_hotwords: GetHotwords
    ) -> typing.Union[Hotwords, HotwordError]:
        """Report available hotwords"""
        try:
            if self.model_dirs:
                # Add all models from model dirs
                model_paths = []
                for model_dir in self.model_dirs:
                    if not model_dir.is_dir():
                        _LOGGER.warning("Model directory missing: %s", str(model_dir))
                        continue

                    for model_file in model_dir.iterdir():
                        if model_file.is_file() and (
                            model_file.suffix in [".umdl", ".pmdl"]
                        ):
                            model_paths.append(model_file)
            else:
                # Add current model(s) only
                model_paths = [Path(model.model_path) for model in self.models]

            hotword_models: typing.List[Hotword] = []
            for model_path in model_paths:
                model_words = " ".join(model_path.with_suffix("").name.split("_"))
                model_type = "universal" if model_path.suffix == ".umdl" else "personal"

                hotword_models.append(
                    Hotword(
                        modelId=model_path.name,
                        modelWords=model_words,
                        modelType=model_type,
                    )
                )

            return Hotwords(
                models={m.modelId: m for m in hotword_models},
                id=get_hotwords.id,
                siteId=get_hotwords.siteId,
            )

        except Exception as e:
            _LOGGER.exception("handle_get_hotwords")
            return HotwordError(
                error=str(e), context=str(get_hotwords), siteId=get_hotwords.siteId
            )

    def detection_thread_proc(self):
        """Handle WAV audio chunks."""
        try:
            while True:
                wav_bytes, siteId = self.wav_queue.get()

                if not self.detectors:
                    self.load_detectors()

                # Extract/convert audio data
                audio_data = self.maybe_convert_wav(wav_bytes)

                # Add to persistent buffer
                self.audio_buffer += audio_data

                # Process in chunks.
                # Any remaining audio data will be kept in buffer.
                while len(self.audio_buffer) >= self.chunk_size:
                    chunk = self.audio_buffer[: self.chunk_size]
                    self.audio_buffer = self.audio_buffer[self.chunk_size :]

                    for detector_index, detector in enumerate(self.detectors):
                        # Return is:
                        # -2 silence
                        # -1 error
                        #  0 voice
                        #  n index n-1
                        result_index = detector.RunDetection(chunk)

                        if result_index > 0:
                            # Detection
                            if detector_index < len(self.wakeword_ids):
                                wakewordId = self.wakeword_ids[detector_index]
                            else:
                                wakewordId = "default"

                            message = self.handle_detection(
                                detector_index, siteId=siteId
                            )
                            self.publish(message, wakewordId=wakewordId)
        except Exception:
            _LOGGER.exception("detection_thread_proc")

    # -------------------------------------------------------------------------

    def udp_thread_proc(self):
        """Handle WAV chunks from UDP socket."""
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.bind(("127.0.0.1", self.udp_audio_port))
            _LOGGER.debug("Listening for audio on UDP port %s", self.udp_audio_port)

            while True:
                wav_bytes, _ = udp_socket.recvfrom(
                    self.udp_chunk_size + WAV_HEADER_BYTES
                )
                self.wav_queue.put((wav_bytes, self.udp_siteId))
        except Exception:
            _LOGGER.exception("udp_thread_proc")

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            # Start threads
            threading.Thread(target=self.detection_thread_proc, daemon=True).start()

            if self.udp_audio_port is not None:
                threading.Thread(target=self.udp_thread_proc, daemon=True).start()

            topics = [
                HotwordToggleOn.topic(),
                HotwordToggleOff.topic(),
                GetHotwords.topic(),
            ]

            if self.audioframe_topics:
                # Specific siteIds
                topics.extend(self.audioframe_topics)
            else:
                # All siteIds
                topics.append(AudioFrame.topic(siteId="+"))

            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            if not msg.topic.endswith("/audioFrame"):
                _LOGGER.debug("Received %s byte(s) on %s", len(msg.payload), msg.topic)

            # Check enable/disable messages
            if msg.topic == HotwordToggleOn.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.enabled = True
                    self.first_audio = True
                    _LOGGER.debug("Enabled")
            elif msg.topic == HotwordToggleOff.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.enabled = False
                    _LOGGER.debug("Disabled")
            elif self.enabled and AudioFrame.is_topic(msg.topic):
                # Handle audio frames
                if (not self.audioframe_topics) or (
                    msg.topic in self.audioframe_topics
                ):
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    siteId = AudioFrame.get_siteId(msg.topic)
                    self.handle_audio_frame(msg.payload, siteId=siteId)
            elif msg.topic == GetHotwords.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.publish(
                        self.handle_get_hotwords(Hotwords.from_dict(json_payload))
                    )

        except Exception:
            _LOGGER.exception("on_message")

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            _LOGGER.debug("-> %s", message)
            topic = message.topic(**topic_args)
            payload = json.dumps(attr.asdict(message))
            _LOGGER.debug("Publishing %s char(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    # -------------------------------------------------------------------------

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        if self.siteIds:
            return json_payload.get("siteId", "default") in self.siteIds

        # All sites
        return True

    # -------------------------------------------------------------------------

    def _convert_wav(self, wav_bytes: bytes) -> bytes:
        """Converts WAV data to required format with sox. Return raw audio."""
        return subprocess.run(
            [
                "sox",
                "-t",
                "wav",
                "-",
                "-r",
                str(self.sample_rate),
                "-e",
                "signed-integer",
                "-b",
                str(self.sample_width * 8),
                "-c",
                str(self.channels),
                "-t",
                "raw",
                "-",
            ],
            check=True,
            stdout=subprocess.PIPE,
            input=wav_bytes,
        ).stdout

    def maybe_convert_wav(self, wav_bytes: bytes) -> bytes:
        """Converts WAV data to required format if necessary. Returns raw audio."""
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io, "rb") as wav_file:
                if (
                    (wav_file.getframerate() != self.sample_rate)
                    or (wav_file.getsampwidth() != self.sample_width)
                    or (wav_file.getnchannels() != self.channels)
                ):
                    # Return converted wav
                    return self._convert_wav(wav_bytes)

                # Return original audio
                return wav_file.readframes(wav_file.getnframes())
