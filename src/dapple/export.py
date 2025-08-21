"""
Export functionality for converting SVG to various formats using Inkscape.
"""

import subprocess
import tempfile
import os
from typing import Union, Optional, BinaryIO, Dict, Any, List
from pathlib import Path
from enum import Enum


class ExportError(Exception):
    """Exception raised when export operations fail."""
    pass


class ExportFormat(Enum):
    """Supported export formats."""
    PNG = "png"
    PDF = "pdf"
    PS = "ps"
    EPS = "eps"
    SVG = "svg"  # Can be used for SVG optimization/processing
    EMF = "emf"
    WMF = "wmf"
    XAML = "xaml"


class InkscapeExporter:
    """Generic Inkscape exporter that handles various output formats."""

    def __init__(self):
        self._inkscape_path = None

    @property
    def inkscape_path(self) -> str:
        """Get the path to Inkscape executable, checking only once."""
        if self._inkscape_path is None:
            self._inkscape_path = self._find_inkscape()
        return self._inkscape_path

    def _find_inkscape(self) -> str:
        """
        Check if Inkscape is available and return its path.

        Returns:
            Path to inkscape executable

        Raises:
            ExportError: If Inkscape is not found
        """
        try:
            # Try to find inkscape in PATH
            result = subprocess.run(
                ['which', 'inkscape'],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                return result.stdout.strip()

            # Try common locations
            common_paths = [
                '/usr/bin/inkscape',
                '/usr/local/bin/inkscape',
                '/snap/bin/inkscape',
                '/Applications/Inkscape.app/Contents/MacOS/inkscape',
            ]

            for path in common_paths:
                if os.path.exists(path):
                    return path

            raise ExportError(
                "Inkscape not found. Please install Inkscape:\n"
                "  Ubuntu/Debian: sudo apt-get install inkscape\n"
                "  Fedora: sudo dnf install inkscape\n"
                "  Arch: sudo pacman -S inkscape\n"
                "  macOS: brew install --cask inkscape\n"
                "  Or download from: https://inkscape.org/release/"
            )

        except Exception as e:
            raise ExportError(f"Error checking for Inkscape: {e}")

    def export(self,
               svg_content: str,
               format: Union[ExportFormat, str],
               output: Union[str, BinaryIO, None] = None,
               options: Optional[Dict[str, Any]] = None) -> Optional[bytes]:
        """
        Generic export method that handles conversion to various formats.

        Args:
            svg_content: The SVG XML content as a string
            format: The output format (ExportFormat enum or string)
            output: Output destination - can be:
                - A string path to write the file
                - A file-like object opened in binary mode
                - None to return the data as bytes
            options: Format-specific options (e.g., dpi, width, height, text_to_path)

        Returns:
            Data as bytes if output is None, otherwise None

        Raises:
            ExportError: If Inkscape is not available or conversion fails
        """
        # Normalize format
        if isinstance(format, str):
            try:
                format = ExportFormat(format.lower())
            except ValueError:
                raise ExportError(f"Unsupported format: {format}")

        # Build command based on format and options
        cmd = self._build_command(format, options or {})

        # Handle different output types
        if output is None:
            return self._export_to_bytes(svg_content, cmd)
        elif isinstance(output, str):
            return self._export_to_file(svg_content, cmd, output)
        else:
            return self._export_to_stream(svg_content, cmd, output)

    def _build_command(self, format: ExportFormat, options: Dict[str, Any]) -> List[str]:
        """Build the Inkscape command line arguments."""
        cmd = [self.inkscape_path, '--pipe', f'--export-type={format.value}']

        # Add format-specific options
        if format == ExportFormat.PNG:
            # PNG-specific options
            if 'dpi' in options:
                cmd.append(f'--export-dpi={options["dpi"]}')
            if 'width' in options:
                cmd.append(f'--export-width={options["width"]}')
            if 'height' in options:
                cmd.append(f'--export-height={options["height"]}')
            if 'background' in options:
                cmd.append(f'--export-background={options["background"]}')
            if 'background_opacity' in options:
                cmd.append(f'--export-background-opacity={options["background_opacity"]}')

        elif format == ExportFormat.PDF:
            # PDF-specific options
            if options.get('text_to_path', False):
                cmd.append('--export-text-to-path')
            if 'pdf_version' in options:
                cmd.append(f'--export-pdf-version={options["pdf_version"]}')

        elif format in [ExportFormat.PS, ExportFormat.EPS]:
            # PostScript/EPS options
            if options.get('text_to_path', False):
                cmd.append('--export-text-to-path')
            if 'ps_level' in options:
                cmd.append(f'--export-ps-level={options["ps_level"]}')

        # Common options that apply to multiple formats
        if 'area' in options:
            # Options: page, drawing, or x0:y0:x1:y1
            cmd.append(f'--export-area={options["area"]}')

        if 'margin' in options:
            cmd.append(f'--export-margin={options["margin"]}')

        if 'id' in options:
            # Export only the object with this ID
            cmd.append(f'--export-id={options["id"]}')

        if options.get('use_hints', False):
            cmd.append('--export-use-hints')

        return cmd

    def _export_to_bytes(self, svg_content: str, cmd: List[str]) -> bytes:
        """Export to stdout and return as bytes."""
        cmd.append('--export-filename=-')

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate(svg_content.encode('utf-8'))

        if process.returncode != 0:
            raise ExportError(f"Inkscape export failed: {stderr.decode('utf-8')}")

        return stdout

    def _export_to_file(self, svg_content: str, cmd: List[str], output_path: str) -> None:
        """Export directly to a file."""
        cmd.append(f'--export-filename={output_path}')

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        _, stderr = process.communicate(svg_content.encode('utf-8'))

        if process.returncode != 0:
            raise ExportError(f"Inkscape export failed: {stderr.decode('utf-8')}")

    def _export_to_stream(self, svg_content: str, cmd: List[str], stream: BinaryIO) -> None:
        """Export to a file-like object via temporary file."""
        # We need to use a temporary file since Inkscape can't always write to stdout reliably
        # Get the format from the command to determine the appropriate extension
        export_type = None
        for arg in cmd:
            if arg.startswith('--export-type='):
                export_type = arg.split('=')[1]
                break

        suffix = f'.{export_type}' if export_type else '.tmp'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd.append(f'--export-filename={tmp_path}')

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            _, stderr = process.communicate(svg_content.encode('utf-8'))

            if process.returncode != 0:
                raise ExportError(f"Inkscape export failed: {stderr.decode('utf-8')}")

            # Read the temporary file and write to the stream
            with open(tmp_path, 'rb') as f:
                stream.write(f.read())

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# Create a singleton exporter instance
_exporter = InkscapeExporter()


def export_svg(svg_content: str,
               format: Union[ExportFormat, str],
               output: Union[str, BinaryIO, None] = None,
               **options) -> Optional[bytes]:
    """
    Export SVG content to various formats using Inkscape.

    Args:
        svg_content: The SVG XML content as a string
        format: The output format (ExportFormat enum or string like 'png', 'pdf', etc.)
        output: Output destination - can be:
            - A string path to write the file
            - A file-like object opened in binary mode
            - None to return the data as bytes
        **options: Format-specific options:
            Common options:
                - area: Export area ('page', 'drawing', or 'x0:y0:x1:y1')
                - margin: Margin around the export area
                - id: Export only the object with this ID
                - use_hints: Use export hints from the SVG file
            PNG options:
                - dpi: Resolution in dots per inch (default: 96)
                - width: Width in pixels
                - height: Height in pixels
                - background: Background color
                - background_opacity: Background opacity (0.0 to 1.0)
            PDF/PS/EPS options:
                - text_to_path: Convert text to paths (default: False)
                - pdf_version: PDF version (e.g., '1.4', '1.5')
                - ps_level: PostScript level (2 or 3)

    Returns:
        Data as bytes if output is None, otherwise None

    Raises:
        ExportError: If Inkscape is not available or conversion fails
    """
    return _exporter.export(svg_content, format, output, options)


# Convenience functions for backward compatibility and common use cases

def svg_to_png(svg_content: str,
               output: Union[str, BinaryIO, None] = None,
               dpi: int = 96,
               width: Optional[int] = None,
               height: Optional[int] = None,
               background: Optional[str] = None,
               background_opacity: Optional[float] = None) -> Optional[bytes]:
    """
    Convert SVG content to PNG using Inkscape.

    Args:
        svg_content: The SVG XML content as a string
        output: Output destination
        dpi: Resolution in dots per inch (default: 96)
        width: Optional width in pixels
        height: Optional height in pixels
        background: Optional background color
        background_opacity: Optional background opacity (0.0 to 1.0)

    Returns:
        PNG data as bytes if output is None, otherwise None
    """
    options = {'dpi': dpi}
    if width is not None:
        options['width'] = width
    if height is not None:
        options['height'] = height
    if background is not None:
        options['background'] = background
    if background_opacity is not None:
        options['background_opacity'] = background_opacity

    return export_svg(svg_content, ExportFormat.PNG, output, **options)


def svg_to_pdf(svg_content: str,
               output: Union[str, BinaryIO, None] = None,
               text_to_path: bool = False,
               pdf_version: Optional[str] = None) -> Optional[bytes]:
    """
    Convert SVG content to PDF using Inkscape.

    Args:
        svg_content: The SVG XML content as a string
        output: Output destination
        text_to_path: Convert text to paths (default: False)
        pdf_version: PDF version (e.g., '1.4', '1.5')

    Returns:
        PDF data as bytes if output is None, otherwise None
    """
    options = {'text_to_path': text_to_path}
    if pdf_version is not None:
        options['pdf_version'] = pdf_version

    return export_svg(svg_content, ExportFormat.PDF, output, **options)


def svg_to_eps(svg_content: str,
               output: Union[str, BinaryIO, None] = None,
               text_to_path: bool = False,
               ps_level: Optional[int] = None) -> Optional[bytes]:
    """
    Convert SVG content to EPS using Inkscape.

    Args:
        svg_content: The SVG XML content as a string
        output: Output destination
        text_to_path: Convert text to paths (default: False)
        ps_level: PostScript level (2 or 3)

    Returns:
        EPS data as bytes if output is None, otherwise None
    """
    options = {'text_to_path': text_to_path}
    if ps_level is not None:
        options['ps_level'] = ps_level

    return export_svg(svg_content, ExportFormat.EPS, output, **options)


def svg_to_ps(svg_content: str,
              output: Union[str, BinaryIO, None] = None,
              text_to_path: bool = False,
              ps_level: Optional[int] = None) -> Optional[bytes]:
    """
    Convert SVG content to PostScript using Inkscape.

    Args:
        svg_content: The SVG XML content as a string
        output: Output destination
        text_to_path: Convert text to paths (default: False)
        ps_level: PostScript level (2 or 3)

    Returns:
        PostScript data as bytes if output is None, otherwise None
    """
    options = {'text_to_path': text_to_path}
    if ps_level is not None:
        options['ps_level'] = ps_level

    return export_svg(svg_content, ExportFormat.PS, output, **options)
