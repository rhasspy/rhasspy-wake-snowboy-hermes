"""Hermes MQTT server for Rhasspy wakeword with snowboy"""
import asyncio
import logging
import queue
import socket
import threading
import typing
from pathlib import Path

import attr
from rhasspyhermes.audioserver import AudioFrame
from rhasspyhermes.base import Message
from rhasspyhermes.client import HermesClient, TopicArgs
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

    def float_sensitivity(self) -> float:
        """Get float of first sensitivity value."""
        # 0.5,0.5
        return float(self.sensitivity.split(",")[0])


# -----------------------------------------------------------------------------


class WakeHermesMqtt(HermesClient):
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
        loop=None,
    ):
        super().__init__(
            "rhasspywake_snowboy_hermes",
            client,
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
            siteIds=siteIds,
            loop=loop,
        )

        self.subscribe(AudioFrame, HotwordToggleOn, HotwordToggleOff, GetHotwords)

        self.models = models
        self.wakeword_ids = wakeword_ids
        self.model_dirs = model_dirs or []

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
        self.udp_siteId = self.siteId

        self.first_audio: bool = True

        self.audio_buffer = bytes()

        # Load detector
        self.detectors: typing.List[typing.Any] = []
        self.model_ids: typing.List[str] = []

        # Event loop
        self.loop = loop or asyncio.get_event_loop()

        # Start threads
        threading.Thread(target=self.detection_thread_proc, daemon=True).start()

        if self.udp_audio_port is not None:
            threading.Thread(target=self.udp_thread_proc, daemon=True).start()

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

    async def handle_audio_frame(self, wav_bytes: bytes, siteId: str = "default"):
        """Process a single audio frame"""
        self.wav_queue.put((wav_bytes, siteId))

    async def handle_detection(
        self, model_index: int, wakewordId: str, siteId: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[HotwordDetected, TopicArgs], HotwordError]
    ]:
        """Handle a successful hotword detection"""
        try:
            assert len(self.model_ids) > model_index, f"Missing {model_index} in models"

            yield (
                HotwordDetected(
                    siteId=siteId,
                    modelId=self.model_ids[model_index],
                    currentSensitivity=self.models[model_index].float_sensitivity(),
                    modelVersion="",
                    modelType="personal",
                ),
                {"wakewordId": wakewordId},
            )
        except Exception as e:
            _LOGGER.exception("handle_detection")
            yield HotwordError(error=str(e), context=str(model_index), siteId=siteId)

    async def handle_get_hotwords(
        self, get_hotwords: GetHotwords
    ) -> typing.AsyncIterable[typing.Union[Hotwords, HotwordError]]:
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

            yield Hotwords(
                models={m.modelId: m for m in hotword_models},
                id=get_hotwords.id,
                siteId=get_hotwords.siteId,
            )

        except Exception as e:
            _LOGGER.exception("handle_get_hotwords")
            yield HotwordError(
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

                            asyncio.run_coroutine_threadsafe(
                                self.publish_all(
                                    self.handle_detection(
                                        detector_index, wakewordId, siteId=siteId
                                    )
                                ),
                                self.loop,
                            )
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

    async def on_message(
        self,
        message: Message,
        siteId: typing.Optional[str] = None,
        sessionId: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ):
        """Received message from MQTT broker."""
        # Check enable/disable messages
        if isinstance(message, HotwordToggleOn):
            self.enabled = True
            self.first_audio = True
            _LOGGER.debug("Enabled")
        elif isinstance(message, HotwordToggleOff):
            self.enabled = False
            _LOGGER.debug("Disabled")
        elif isinstance(message, AudioFrame):
            if self.enabled:
                assert siteId, "Missing siteId"
                await self.handle_audio_frame(message.wav_bytes, siteId=siteId)
        elif isinstance(message, GetHotwords):
            await self.publish_all(self.handle_get_hotwords(message))
