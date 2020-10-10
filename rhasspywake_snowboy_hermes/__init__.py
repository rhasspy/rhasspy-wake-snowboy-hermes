"""Hermes MQTT server for Rhasspy wakeword with snowboy"""
import asyncio
import logging
import queue
import socket
import threading
import typing
from dataclasses import dataclass
from pathlib import Path

from rhasspyhermes.audioserver import AudioFrame
from rhasspyhermes.base import Message
from rhasspyhermes.client import GeneratorType, HermesClient, TopicArgs
from rhasspyhermes.wake import (
    GetHotwords,
    Hotword,
    HotwordDetected,
    HotwordError,
    Hotwords,
    HotwordToggleOff,
    HotwordToggleOn,
    HotwordToggleReason,
)

WAV_HEADER_BYTES = 44
_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


@dataclass
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
        site_ids: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        chunk_size: int = 960,
        udp_audio: typing.Optional[typing.List[typing.Tuple[str, int, str]]] = None,
        udp_chunk_size: int = 2048,
    ):
        super().__init__(
            "rhasspywake_snowboy_hermes",
            client,
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
            site_ids=site_ids,
        )

        self.subscribe(AudioFrame, HotwordToggleOn, HotwordToggleOff, GetHotwords)

        self.models = models
        self.wakeword_ids = wakeword_ids
        self.model_dirs = model_dirs or []

        self.enabled = enabled
        self.disabled_reasons: typing.Set[str] = set()

        # Required audio format
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels

        self.chunk_size = chunk_size

        # Queue of WAV audio chunks to process (plus site_id)
        self.wav_queue: queue.Queue = queue.Queue()

        self.first_audio: bool = True
        self.audio_buffer = bytes()

        # Load detector
        self.detectors: typing.List[typing.Any] = []
        self.model_ids: typing.List[str] = []

        # Start threads
        threading.Thread(target=self.detection_thread_proc, daemon=True).start()

        # Listen for raw audio on UDP too
        self.udp_chunk_size = udp_chunk_size

        if udp_audio:
            for udp_host, udp_port, udp_site_id in udp_audio:
                threading.Thread(
                    target=self.udp_thread_proc,
                    args=(udp_host, udp_port, udp_site_id),
                    daemon=True,
                ).start()

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

    async def handle_audio_frame(self, wav_bytes: bytes, site_id: str = "default"):
        """Process a single audio frame"""
        self.wav_queue.put((wav_bytes, site_id))

    async def handle_detection(
        self, model_index: int, wakeword_id: str, site_id: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[HotwordDetected, TopicArgs], HotwordError]
    ]:
        """Handle a successful hotword detection"""
        try:
            assert len(self.model_ids) > model_index, f"Missing {model_index} in models"
            sensitivity = 0.5

            if model_index < len(self.models):
                sensitivity = self.models[model_index].float_sensitivity()

            yield (
                HotwordDetected(
                    site_id=site_id,
                    model_id=self.model_ids[model_index],
                    current_sensitivity=sensitivity,
                    model_version="",
                    model_type="personal",
                ),
                {"wakeword_id": wakeword_id},
            )
        except Exception as e:
            _LOGGER.exception("handle_detection")
            yield HotwordError(error=str(e), context=str(model_index), site_id=site_id)

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
                        model_id=model_path.name,
                        model_words=model_words,
                        model_type=model_type,
                    )
                )

            yield Hotwords(
                models=hotword_models, id=get_hotwords.id, site_id=get_hotwords.site_id
            )

        except Exception as e:
            _LOGGER.exception("handle_get_hotwords")
            yield HotwordError(
                error=str(e), context=str(get_hotwords), site_id=get_hotwords.site_id
            )

    def detection_thread_proc(self):
        """Handle WAV audio chunks."""
        try:
            while True:
                wav_bytes, site_id = self.wav_queue.get()

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
                                wakeword_id = self.wakeword_ids[detector_index]
                            else:
                                wakeword_id = ""

                            if not wakeword_id:
                                if detector_index < len(self.models):
                                    # Use file name
                                    wakeword_id = self.models[
                                        detector_index
                                    ].model_path.stem
                                else:
                                    # Fall back to default
                                    wakeword_id = "default"

                            _LOGGER.debug(
                                "Wake word detected: %s (site_id=%s)",
                                wakeword_id,
                                site_id,
                            )

                            asyncio.run_coroutine_threadsafe(
                                self.publish_all(
                                    self.handle_detection(
                                        detector_index, wakeword_id, site_id=site_id
                                    )
                                ),
                                self.loop,
                            )
        except Exception:
            _LOGGER.exception("detection_thread_proc")

    # -------------------------------------------------------------------------

    def udp_thread_proc(self, host: str, port: int, site_id: str):
        """Handle WAV chunks from UDP socket."""
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.bind((host, port))
            _LOGGER.debug("Listening for audio on UDP %s:%s", host, port)

            while True:
                wav_bytes, _ = udp_socket.recvfrom(
                    self.udp_chunk_size + WAV_HEADER_BYTES
                )

                if self.enabled:
                    self.wav_queue.put((wav_bytes, site_id))
        except Exception:
            _LOGGER.exception("udp_thread_proc")

    # -------------------------------------------------------------------------

    async def on_message_blocking(
        self,
        message: Message,
        site_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ) -> GeneratorType:
        """Received message from MQTT broker."""
        # Check enable/disable messages
        if isinstance(message, HotwordToggleOn):
            if message.reason == HotwordToggleReason.UNKNOWN:
                # Always enable on unknown
                self.disabled_reasons.clear()
            else:
                self.disabled_reasons.discard(message.reason)

            if self.disabled_reasons:
                _LOGGER.debug("Still disabled: %s", self.disabled_reasons)
            else:
                self.enabled = True
                self.first_audio = True
                _LOGGER.debug("Enabled")
        elif isinstance(message, HotwordToggleOff):
            self.enabled = False
            self.disabled_reasons.add(message.reason)
            _LOGGER.debug("Disabled")
        elif isinstance(message, AudioFrame):
            if self.enabled:
                assert site_id, "Missing site_id"
                await self.handle_audio_frame(message.wav_bytes, site_id=site_id)
        elif isinstance(message, GetHotwords):
            async for hotword_result in self.handle_get_hotwords(message):
                yield hotword_result
        else:
            _LOGGER.warning("Unexpected message: %s", message)
