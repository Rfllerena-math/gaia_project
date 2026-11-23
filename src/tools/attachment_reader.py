"""Tool for inspecting local attachments supplied with a GAIA question."""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from pathlib import Path

import requests
from langchain_core.tools import tool

from .file_tools import read_data_file
from .file_reader import ROOT
from .video_pipeline import _extract_audio, _sample_frames, _transcribe_audio

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")


def _describe_image(image_path: Path, question: str) -> str:
    """Use the configured local vision model to inspect an image attachment."""
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={
            "model": OLLAMA_VISION_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Describe the attachment accurately, focusing on details needed to "
                        f"answer this question: {question or 'Describe the attachment.'}"
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


def _inspect_video(video_path: Path, question: str) -> str:
    """Return a transcript and representative visual observations for a local video."""
    with tempfile.TemporaryDirectory(prefix="gaia-attachment-") as directory_name:
        directory = Path(directory_name)
        try:
            audio_path = _extract_audio(video_path, directory)
            transcript = _transcribe_audio(audio_path)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            transcript = f"Audio extraction failed: {error}"

        descriptions = []
        try:
            for frame_path in _sample_frames(video_path, directory, max_frames=8):
                description = _describe_image(frame_path, question)
                if description:
                    descriptions.append(description)
        except requests.RequestException as error:
            descriptions.append(f"Visual analysis failed: {error}")

    visuals = "\n".join(f"- {description}" for description in descriptions)
    return f"Transcript:\n{transcript}\n\nVisual observations:\n{visuals or 'None captured.'}"


@tool
def inspect_attachment(path: str, question: str = "") -> str:
    """Inspect a local attachment in this project: image, audio, video, or document/data file.

    Give the path relative to the project root and provide the user's question to focus image or
    video observations. Images use the local Ollama vision model. Audio is transcribed with the
    optional faster-whisper dependency. Videos produce both a transcript and frame observations.
    """
    requested_path = (ROOT / path).resolve()
    if not requested_path.is_relative_to(ROOT):
        return "Only files within the project root can be inspected."
    if not requested_path.is_file():
        return f"File not found: {path}"

    suffix = requested_path.suffix.lower()
    try:
        if suffix in IMAGE_EXTENSIONS:
            return _describe_image(requested_path, question)
        if suffix in AUDIO_EXTENSIONS:
            return _transcribe_audio(requested_path)
        if suffix in VIDEO_EXTENSIONS:
            return _inspect_video(requested_path, question)
        return read_data_file.invoke({"path": path})
    except (OSError, requests.RequestException) as error:
        return f"Unable to inspect {path}: {error}"