"""Audio-transcript plus visual-observation pipeline for general video questions."""

from __future__ import annotations

import base64
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
from langchain_core.tools import tool

from .video_analyzer import _is_supported_video_url

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
MAX_ANALYZED_FRAMES = 12


def _download_video(url: str, directory: Path) -> Path:
    """Download a public YouTube video at a moderate resolution."""
    video_path = directory / "video.mp4"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--format",
            "mp4[height<=480]/best[height<=480]/best",
            "--output",
            str(video_path),
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return video_path


def _extract_audio(video_path: Path, directory: Path) -> Path:
    """Extract a mono 16kHz WAV track from the video for transcription."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg is not installed or not on PATH")
    audio_path = directory / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", str(audio_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return audio_path


def _transcribe_audio(audio_path: Path) -> str:
    """Transcribe the extracted audio track using a local faster-whisper model."""
    try:
        whisper_module = importlib.import_module("faster_whisper")
        whisper_model = whisper_module.WhisperModel
    except ImportError:
        return "Audio transcription requires the optional 'faster-whisper' package."
    try:
        model = whisper_model(WHISPER_MODEL_SIZE)
        segments, _ = model.transcribe(str(audio_path))
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:
        return f"Audio transcription failed: {error}"
    return transcript or "No speech detected."


def _sample_frames(video_path: Path, directory: Path, max_frames: int) -> list[Path]:
    """Sample evenly spaced frames across the video for visual analysis."""
    try:
        import cv2
    except ImportError:
        return []
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or frame_count <= 0:
        capture.release()
        return []
    step = max(1, frame_count // max_frames)
    frame_paths: list[Path] = []
    for index, frame_number in enumerate(range(0, frame_count, step)):
        if index >= max_frames:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            continue
        frame_path = directory / f"frame-{index:03d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
    capture.release()
    return frame_paths


def _describe_frame(frame_path: Path, question: str) -> str:
    """Ask the local vision model for observations relevant to the question."""
    encoded_image = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    response = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={
            "model": OLLAMA_VISION_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Describe only the visible details in this video frame relevant to the "
                        f"following question. Be factual and concise.\nQuestion: {question}"
                    ),
                    "images": [encoded_image],
                }
            ],
            "options": {"temperature": 0},
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "").strip()


@tool
def analyze_video(url: str, question: str) -> str:
    """Download a YouTube video and return its transcript plus visual observations.

    Extracts audio, transcribed with faster-whisper, and samples frames described by the
    local Ollama vision model. Requires yt-dlp, ffmpeg, and OpenCV; faster-whisper is optional
    and only needed for the audio transcript.
    """
    if not _is_supported_video_url(url):
        return "Video analysis supports public YouTube URLs only."

    with tempfile.TemporaryDirectory(prefix="gaia-video2-") as directory_name:
        directory = Path(directory_name)
        try:
            video_path = _download_video(url, directory)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            return f"Unable to download video: {error}"

        try:
            audio_path = _extract_audio(video_path, directory)
            transcript = _transcribe_audio(audio_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            transcript = f"Audio extraction failed: {error}"

        observations: list[str] = []
        try:
            for frame_path in _sample_frames(video_path, directory, MAX_ANALYZED_FRAMES):
                description = _describe_frame(frame_path, question)
                if description:
                    observations.append(description)
        except requests.RequestException as error:
            observations.append(f"Visual analysis failed: {error}")

    visual_observations = (
        "\n".join(f"- {item}" for item in observations)
        or "No visual observations captured."
    )
    return f"Transcript:\n{transcript}\n\nVisual observations:\n{visual_observations}"
