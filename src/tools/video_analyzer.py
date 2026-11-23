"""Video-frame analysis tools for visual GAIA questions."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
MAX_VIDEO_FRAMES = 120


class VideoAnalysisAgent:
    """Specialized local vision agent for evidence from YouTube video frames."""

    def analyze_bird_species(
        self, url: str, sample_interval_seconds: float = 1.0
    ) -> str:
        """Delegate video-frame recognition to the Qwen vision-backed tool."""
        return analyze_video_bird_species.invoke(
            {"url": url, "sample_interval_seconds": sample_interval_seconds}
        )


def _is_supported_video_url(url: str) -> bool:
    """Accept public YouTube URLs, which yt-dlp can retrieve consistently."""
    hostname = urlparse(url).hostname or ""
    return hostname.lower() in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def extract_youtube_url(text: str) -> str | None:
    """Return the first public YouTube URL in a request, if present."""
    match = re.search(r"https?://[^\s)>\]]+", text)
    if match and _is_supported_video_url(match.group(0)):
        return match.group(0)
    return None


def _extract_species(response: str) -> list[str]:
    """Read the vision model's JSON result, returning no species on malformed output."""
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        return []
    species = result.get("species", [])
    if not isinstance(species, list) or not all(isinstance(item, str) for item in species):
        return []
    return sorted({item.strip() for item in species if item.strip()})


def _analyze_frame(frame_path: Path) -> list[str]:
    """Use Ollama's vision endpoint to identify distinct bird species in one frame."""
    encoded_image = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    response = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={
            "model": OLLAMA_VISION_MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Identify the distinct bird species visibly on camera in this video frame. "
                        "Count a species once even if several birds of it appear. Do not guess from "
                        "context; include only species supported by visible features. Reply only as "
                        'JSON: {"species": ["common name", ...]}.'
                    ),
                    "images": [encoded_image],
                }
            ],
            "options": {"temperature": 0},
        },
        timeout=90,
    )
    response.raise_for_status()
    return _extract_species(response.json().get("message", {}).get("content", ""))


@tool
def analyze_video_bird_species(url: str, sample_interval_seconds: float = 1.0) -> str:
    """Analyze a public YouTube video and report the maximum distinct bird species visible in one frame.

    Requires yt-dlp, OpenCV, and a running Ollama vision model. Set OLLAMA_VISION_MODEL to the
    installed vision model name. Reduce sample_interval_seconds for short, fast-changing videos.
    """
    if not _is_supported_video_url(url):
        return "Video analysis supports public YouTube URLs only."
    if sample_interval_seconds <= 0:
        return "sample_interval_seconds must be greater than zero."

    try:
        import cv2
    except ImportError:
        return "Video analysis requires opencv-python-headless. Install dependencies with pip install -r requirements.txt."

    with tempfile.TemporaryDirectory(prefix="gaia-video-") as directory:
        video_path = Path(directory) / "video.mp4"
        try:
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
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            return f"Unable to download video for analysis: {error}"

        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or frame_count <= 0:
            capture.release()
            return "Unable to decode video frames."

        step = max(1, round(fps * sample_interval_seconds))
        sampled_indices = range(0, frame_count, step)
        findings: list[tuple[float, list[str]]] = []
        try:
            for index, frame_number in enumerate(sampled_indices):
                if index >= MAX_VIDEO_FRAMES:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = capture.read()
                if not success:
                    continue
                frame_path = Path(directory) / f"frame-{index:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                species = _analyze_frame(frame_path)
                if species:
                    findings.append((frame_number / fps, species))
        except requests.RequestException as error:
            return f"Vision model request failed: {error}"
        finally:
            capture.release()

    if not findings:
        return "No bird species could be identified in the sampled frames."
    maximum_time, maximum_species = max(findings, key=lambda finding: len(finding[1]))
    evidence = "; ".join(
        f"{timestamp:.1f}s: {', '.join(species)}" for timestamp, species in findings
    )
    return (
        f"Maximum distinct bird species in one sampled frame: {len(maximum_species)} "
        f"at {maximum_time:.1f}s ({', '.join(maximum_species)}). Frame evidence: {evidence}"
    )