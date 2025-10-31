"""
Cross-platform clipboard support with MIME type handling for SVG content.
"""

import os
import subprocess
import sys
from typing import Optional


class ClipboardError(Exception):
    """Exception raised when clipboard operations fail."""

    pass


def _detect_display_server() -> Optional[str]:
    """Detect whether we're running on X11 or Wayland."""
    # Check for Wayland
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    # Check for X11
    if os.environ.get("DISPLAY"):
        return "x11"
    return None


def _copy_x11(content: str, mime_type: str = "image/svg+xml") -> None:
    """Copy content to X11 clipboard with specified MIME type using xclip."""
    try:
        # Try xclip first (more common and supports MIME types well)
        process = subprocess.Popen(
            ["xclip", "-selection", "clipboard", "-t", mime_type],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(content.encode("utf-8"))

        if process.returncode != 0:
            raise ClipboardError(f"xclip failed: {stderr.decode('utf-8')}")

    except FileNotFoundError:
        # Fall back to xsel if xclip is not available
        try:
            # Note: xsel doesn't support MIME types as well, but we try anyway
            process = subprocess.Popen(
                ["xsel", "--clipboard", "--input"],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate(content.encode("utf-8"))

            if process.returncode != 0:
                raise ClipboardError(f"xsel failed: {stderr.decode('utf-8')}")

        except FileNotFoundError:
            raise ClipboardError(
                "Neither xclip nor xsel found. Please install one of them:\n"
                "  Ubuntu/Debian: sudo apt-get install xclip\n"
                "  Fedora: sudo dnf install xclip\n"
                "  Arch: sudo pacman -S xclip"
            )


def _copy_wayland(content: str, mime_type: str = "image/svg+xml") -> None:
    """Copy content to Wayland clipboard with specified MIME type using wl-copy."""
    try:
        subprocess.run(
            ["wl-copy", "--type", mime_type],
            input=content.encode("utf-8"),
            stderr=subprocess.DEVNULL,  # avoid hanging if wl-copy daemonizes
            check=True,
        )

    except FileNotFoundError:
        raise ClipboardError(
            "wl-copy not found. Please install wl-clipboard:\n"
            "  Ubuntu/Debian: sudo apt-get install wl-clipboard\n"
            "  Fedora: sudo dnf install wl-clipboard\n"
            "  Arch: sudo pacman -S wl-clipboard"
        )


def copy_svg(svg_content: str) -> None:
    """
    Copy SVG content to system clipboard with proper MIME type.

    Args:
        svg_content: The SVG XML content as a string

    Raises:
        ClipboardError: If clipboard operation fails or platform is unsupported
    """
    display_server = _detect_display_server()

    if display_server == "x11":
        _copy_x11(svg_content, "image/svg+xml")
    elif display_server == "wayland":
        _copy_wayland(svg_content, "image/svg+xml")
    elif sys.platform == "darwin":
        # TODO: Implement macOS support
        # Could use pbcopy with UTI types or osascript
        raise ClipboardError("macOS clipboard support not yet implemented")
    elif sys.platform == "win32":
        # TODO: Implement Windows support
        # Could use win32clipboard with CF_HTML or custom format
        raise ClipboardError("Windows clipboard support not yet implemented")
    else:
        raise ClipboardError(
            f"Unsupported platform or display server. "
            f"Platform: {sys.platform}, Display: {display_server}"
        )


def copy_with_fallback(content: str, mime_type: str = "image/svg+xml") -> None:
    """
    Copy content to clipboard with MIME type, falling back to text if needed.

    Args:
        content: The content to copy
        mime_type: The MIME type of the content

    Raises:
        ClipboardError: If clipboard operation fails
    """
    try:
        if mime_type == "image/svg+xml":
            copy_svg(content)
        else:
            # For other MIME types, attempt generic copy
            display_server = _detect_display_server()
            if display_server == "x11":
                _copy_x11(content, mime_type)
            elif display_server == "wayland":
                _copy_wayland(content, mime_type)
            else:
                raise ClipboardError(
                    f"Unsupported MIME type {mime_type} on this platform"
                )
    except ClipboardError:
        # If MIME type copy fails, could fall back to plain text
        # but for SVG we want to preserve the MIME type, so re-raise
        raise
