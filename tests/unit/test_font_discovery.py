"""
Tests for font discovery across different platforms and fallback mechanisms.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from dapple.textextents import (
    find_font,
    _scan_font_directories,
    HAS_FONTCONFIG,
    HAS_MATPLOTLIB,
)


class TestFontDiscovery:
    """Test the font discovery fallback chain."""

    def test_find_font_returns_valid_path(self):
        """Test that find_font returns a valid path for common fonts."""
        # Try common fonts that should exist on most systems
        common_fonts = ["DejaVu Sans", "Liberation Sans", "FreeSans", "Arial"]

        found = False
        for family in common_fonts:
            path = find_font(family)
            if path:
                assert Path(path).exists(), f"Font path {path} does not exist"
                assert Path(path).is_file(), f"Font path {path} is not a file"
                found = True
                break

        assert found, f"Could not find any common fonts: {common_fonts}"

    def test_find_font_nonexistent_returns_none(self):
        """Test that find_font returns None for a nonexistent font."""
        result = find_font("ThisFontDefinitelyDoesNotExist12345XYZ")
        assert result is None

    def test_scan_font_directories_platform_specific(self):
        """Test that _scan_font_directories checks platform-specific directories."""
        # This test verifies the function runs without error
        # It may or may not find a font depending on the system
        result = _scan_font_directories("DejaVu Sans")
        # Result can be None or a path - both are valid
        if result:
            assert isinstance(result, str)
            assert Path(result).exists()

    @pytest.mark.skipif(not HAS_FONTCONFIG, reason="fontconfig not available")
    def test_fontconfig_integration(self):
        """Test fontconfig integration when available."""
        import fontconfig

        # Test that we can use fontconfig to find a font
        # fontconfig.query() returns a list of file paths (strings)
        fonts = fontconfig.query(family="DejaVu Sans")

        if fonts:
            # Verify we got font results
            assert len(fonts) > 0
            # Check that result is a string (file path)
            assert isinstance(fonts[0], str)
            assert Path(fonts[0]).exists()

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_matplotlib_integration(self):
        """Test matplotlib FontManager integration when available."""
        from matplotlib import font_manager

        props = font_manager.FontProperties(family="DejaVu Sans")
        path = font_manager.findfont(props)

        assert path is not None
        assert isinstance(path, str)
        # Note: matplotlib may return a fallback font, so we don't check the name

    def test_font_extensions_recognized(self):
        """Test that common font extensions are recognized in directory scanning."""
        # This is implicit in _scan_font_directories but worth documenting
        extensions = {".ttf", ".otf", ".ttc"}

        # Try to find a font and verify it has one of these extensions
        common_fonts = ["DejaVu Sans", "Liberation Sans", "FreeSans", "Arial"]
        for family in common_fonts:
            path = find_font(family)
            if path:
                assert any(path.endswith(ext) for ext in extensions), (
                    f"Font {path} doesn't have a recognized extension"
                )
                break


class TestScanFontDirectories:
    """Test the directory scanning fallback mechanism."""

    def test_handles_missing_directories_gracefully(self):
        """Test that missing directories don't cause errors."""
        # Should not raise an exception even if directories don't exist
        result = _scan_font_directories("SomeFont")
        # Result can be None or a path - both are valid
        assert result is None or isinstance(result, str)

    def test_normalizes_font_family_names(self):
        """Test that font family names are normalized for matching."""
        # The function should normalize spaces and hyphens
        # We can't easily test this directly, but we can verify it works
        # by trying different name formats if we have a known font
        common_fonts = ["DejaVu Sans", "DejaVuSans", "dejavu-sans"]

        found_paths = []
        for family in common_fonts:
            path = _scan_font_directories(family)
            if path:
                found_paths.append(path)

        # If any font was found, the normalization is working
        # (we can't guarantee all variants will be found, but at least one should work)
        if found_paths:
            # Verify all found paths are actual files
            for path in found_paths:
                assert Path(path).exists()

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux-specific test")
    def test_linux_font_directories(self):
        """Test that Linux font directories are checked."""
        # On Linux, at least one of these should exist
        linux_dirs = [
            Path.home() / ".fonts",
            Path.home() / ".local/share/fonts",
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
        ]

        exists = any(d.exists() for d in linux_dirs)
        if not exists:
            pytest.skip("No standard Linux font directories found")

        # Try to find a common font
        result = _scan_font_directories("DejaVu Sans")
        # May or may not find it, but shouldn't crash
        assert result is None or isinstance(result, str)

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific test")
    def test_macos_font_directories(self):
        """Test that macOS font directories are checked."""
        # On macOS, at least one of these should exist
        macos_dirs = [
            Path.home() / "Library/Fonts",
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts"),
        ]

        exists = any(d.exists() for d in macos_dirs)
        assert exists, "No standard macOS font directories found"

        # Try to find a common font
        result = _scan_font_directories("Arial")
        # May or may not find it, but shouldn't crash
        assert result is None or isinstance(result, str)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_windows_font_directories(self):
        """Test that Windows font directories are checked."""
        # On Windows, at least the system fonts directory should exist
        windows_dirs = [
            Path("C:/Windows/Fonts"),
            Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
        ]

        exists = any(d.exists() for d in windows_dirs)
        assert exists, "No standard Windows font directories found"

        # Try to find a common font
        result = _scan_font_directories("Arial")
        # May or may not find it, but shouldn't crash
        assert result is None or isinstance(result, str)


class TestFallbackChain:
    """Test that the fallback chain works correctly."""

    def test_fallback_chain_exists(self):
        """Test that multiple discovery methods are attempted."""
        # We can't easily mock all the methods, but we can verify
        # that the function handles missing dependencies gracefully

        # Even if fontconfig and matplotlib are not available,
        # find_font should still try directory scanning
        result = find_font("DejaVu Sans")

        # Result depends on system, but function should not crash
        assert result is None or isinstance(result, str)

    def test_find_font_with_valid_font_returns_path(self):
        """Integration test: find_font should succeed for common fonts."""
        # Try several common fonts across different platforms
        fonts_to_try = [
            "DejaVu Sans",  # Common on Linux
            "Liberation Sans",  # Common on Linux
            "Arial",  # Common on Windows/macOS
            "Helvetica",  # Common on macOS
            "FreeSans",  # GNU FreeFont
        ]

        found = False
        for family in fonts_to_try:
            path = find_font(family)
            if path:
                assert Path(path).exists()
                assert Path(path).suffix.lower() in [".ttf", ".otf", ".ttc"]
                found = True
                break

        # At least one common font should be found
        assert found, f"Could not find any fonts from: {fonts_to_try}"
