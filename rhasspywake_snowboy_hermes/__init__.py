"""Hermes MQTT server for Rhasspy wakeword with snowboy"""
import asyncio
import io
import logging
import queue
import socket
import threading
import typing
import wave
from dataclasses import dataclass, field
from pathlib import Path

from snowboy import snowboydecoder, snowboydetect

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
_LOGGER = logging.getLogger("rhasspywake_snowboy_hermes")

# -----------------------------------------------------------------------------


@dataclass
class SiteInfo:
    """Self-contained information for a single site"""

    site_id: str
    detection_thread: typing.Optional[threading.Thread] = None
    audio_buffer: bytes = bytes()
    first_audio: bool = True
    model_ids: typing.List[str] = field(default_factory=list)
    detectors: typing.List[snowboydetect.SnowboyDetect] = field(default_factory=list)

    # Queue of (bytes, is_raw)
    wav_queue: "queue.Queue[typing.Tuple[bytes, bool]]" = field(
        default_factory=queue.Queue
    )


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
        udp_raw_audio: typing.Optional[typing.Iterable[str]] = None,
        udp_forward_mqtt: typing.Optional[typing.Iterable[str]] = None,
        lang: typing.Optional[str] = None,
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

        self.site_info: typing.Dict[str, SiteInfo] = {}

        # Create site information for known sites
        for site_id in self.site_ids:
            site_info = SiteInfo(site_id=site_id)

            # Create and start detection thread
            site_info.detection_thread = threading.Thread(
                target=self.detection_thread_proc, daemon=True, args=(site_info,)
            )
            site_info.detection_thread.start()

            self.site_info[site_id] = site_info

        self.lang = lang

        # Listen for raw audio on UDP too
        self.udp_chunk_size = udp_chunk_size

        # Site ids where UDP audio is raw 16Khz, 16-bit mono PCM chunks instead
        # of WAV chunks.
        self.udp_raw_audio = set(udp_raw_audio or [])

        # Site ids where UDP audio should be forward to MQTT after detection.
        self.udp_forward_mqtt = set(udp_forward_mqtt or [])

        if udp_audio:
            for udp_host, udp_port, udp_site_id in udp_audio:
                threading.Thread(
                    target=self.udp_thread_proc,
                    args=(udp_host, udp_port, udp_site_id),
                    daemon=True,
                ).start()

    # -------------------------------------------------------------------------

    def load_detectors(self, site_info: SiteInfo):
        """Load snowboy detectors from models"""
        site_info.model_ids = []
        site_info.detectors = []

        for model in self.models:
            assert model.model_path.is_file(), f"Missing {model.model_path}"
            _LOGGER.debug("Loading snowboy model: %s", model)

            detector = snowboydetect.SnowboyDetect(
                snowboydecoder.RESOURCE_FILE.encode(), str(model.model_path).encode()
            )

            detector.SetSensitivity(model.sensitivity.encode())
            detector.SetAudioGain(model.audio_gain)
            detector.ApplyFrontend(model.apply_frontend)

            site_info.detectors.append(detector)
            site_info.model_ids.append(model.model_path.stem)

    # -------------------------------------------------------------------------
    def stop(self):
        """Stop detection threads."""
        _LOGGER.debug("Stopping detection threads...")

        for site_info in self.site_info.values():
            if site_info.detection_thread is not None:
                site_info.wav_queue.put((None, None))
                site_info.detection_thread.join()
                site_info.detection_thread = None

            site_info.porcupine = None

        _LOGGER.debug("Stopped")

    # -------------------------------------------------------------------------

    async def handle_audio_frame(self, wav_bytes: bytes, site_id: str = "default"):
        """Process a single audio frame"""
        site_info = self.site_info.get(site_id)
        if site_info is None:
            # Create information for new site
            site_info = SiteInfo(site_id=site_id)
            site_info.detection_thread = threading.Thread(
                target=self.detection_thread_proc, daemon=True, args=(site_info,)
            )

            site_info.detection_thread.start()
            self.site_info[site_id] = site_info

        site_info.wav_queue.put((wav_bytes, False))

    async def handle_detection(
        self, model_index: int, wakeword_id: str, site_info: SiteInfo
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[HotwordDetected, TopicArgs], HotwordError]
    ]:
        """Handle a successful hotword detection"""
        try:
            assert (
                len(site_info.model_ids) > model_index
            ), f"Missing {model_index} in models"
            sensitivity = 0.5

            if model_index < len(self.models):
                sensitivity = self.models[model_index].float_sensitivity()

            yield (
                HotwordDetected(
                    site_id=site_info.site_id,
                    model_id=site_info.model_ids[model_index],
                    current_sensitivity=sensitivity,
                    model_version="",
                    model_type="personal",
                    lang=self.lang,
                ),
                {"wakeword_id": wakeword_id},
            )
        except Exception as e:
            _LOGGER.exception("handle_detection")
            yield HotwordError(
                error=str(e), context=str(model_index), site_id=site_info.site_id
            )

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

    def detection_thread_proc(self, site_info: SiteInfo):
        """Handle WAV audio chunks."""
        try:
            while True:
                wav_bytes, is_raw = site_info.wav_queue.get()
                if wav_bytes is None:
                    # Shutdown signal
                    break

                if site_info.first_audio:
                    _LOGGER.debug("Receiving audio %s", site_info.site_id)
                    site_info.first_audio = False

                if not site_info.detectors:
                    self.load_detectors(site_info)

                if is_raw:
                    # Raw audio chunks
                    audio_data = wav_bytes
                else:
                    # WAV chunks
                    audio_data = self.maybe_convert_wav(wav_bytes)

                # Add to persistent buffer
                site_info.audio_buffer += audio_data

                # Process in chunks.
                # Any remaining audio data will be kept in buffer.
                while len(site_info.audio_buffer) >= self.chunk_size:
                    chunk = site_info.audio_buffer[: self.chunk_size]
                    site_info.audio_buffer = site_info.audio_buffer[self.chunk_size :]

                    for detector_index, detector in enumerate(site_info.detectors):
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
                                site_info.site_id,
                            )

                            assert self.loop is not None
                            asyncio.run_coroutine_threadsafe(
                                self.publish_all(
                                    self.handle_detection(
                                        detector_index, wakeword_id, site_info=site_info
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
            site_info = self.site_info[site_id]
            is_raw_audio = site_id in self.udp_raw_audio
            forward_to_mqtt = site_id in self.udp_forward_mqtt

            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.bind((host, port))
            _LOGGER.debug(
                "Listening for audio on UDP %s:%s (siteId=%s, raw=%s)",
                host,
                port,
                site_id,
                is_raw_audio,
            )

            chunk_size = self.udp_chunk_size
            if is_raw_audio:
                chunk_size += WAV_HEADER_BYTES

            while True:
                wav_bytes, _ = udp_socket.recvfrom(chunk_size)

                if self.enabled:
                    site_info.wav_queue.put((wav_bytes, is_raw_audio))
                elif forward_to_mqtt:
                    # When the wake word service is disabled, ASR should be active
                    if is_raw_audio:
                        # Re-package as WAV chunk and publish to MQTT
                        with io.BytesIO() as wav_buffer:
                            wav_file: wave.Wave_write = wave.open(wav_buffer, "wb")
                            with wav_file:
                                wav_file.setframerate(self.sample_rate)
                                wav_file.setsampwidth(self.sample_width)
                                wav_file.setnchannels(self.channels)
                                wav_file.writeframes(wav_bytes)

                            publish_wav_bytes = wav_buffer.getvalue()
                    else:
                        # Use WAV chunk as-is
                        publish_wav_bytes = wav_bytes

                    self.publish(
                        AudioFrame(wav_bytes=publish_wav_bytes),
                        site_id=site_info.site_id,
                    )
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

                # Reset first audio flags
                for site_info in self.site_info.values():
                    site_info.first_audio = True

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
