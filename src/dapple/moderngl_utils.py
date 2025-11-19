import threading
from typing import Optional, Tuple

import moderngl
import numpy as np


class ModernGLContext:
    """
    Singleton ModernGL context manager for rendering operations.
    Uses a headless context for off-screen rendering.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            exceptions = []
            # Try backends in order. None lets moderngl choose (usually good for desktop).
            # 'egl' is good for headless linux. 'osmesa' is software fallback.
            for backend in [None, "egl", "osmesa"]:
                try:
                    if backend is None:
                        self.ctx = moderngl.create_context(standalone=True, require=330)
                    else:
                        self.ctx = moderngl.create_context(
                            standalone=True, require=330, backend=backend
                        )
                    self._initialized = True
                    return
                except Exception as e:
                    exceptions.append(f"{backend or 'auto'}: {e}")

            raise RuntimeError(
                "Failed to initialize ModernGL context. Tried backends: "
                + "; ".join(exceptions)
            )

    def get_context(self) -> moderngl.Context:
        return self.ctx


def create_framebuffer(
    width: int, height: int, samples: int = 0
) -> Tuple[moderngl.Framebuffer, moderngl.Texture]:
    """
    Create a framebuffer for off-screen rendering.

    Args:
        width: Width in pixels
        height: Height in pixels
        samples: Number of samples for multisampling (0 for no multisampling)

    Returns:
        Tuple of (framebuffer, texture)
    """
    ctx = ModernGLContext().get_context()

    if samples > 0:
        # Create multisampled texture
        texture = ctx.texture_2d_multisample((width, height), 4, samples)
        framebuffer = ctx.framebuffer([texture])
    else:
        # Create regular texture
        texture = ctx.texture((width, height), 4)  # RGBA
        framebuffer = ctx.framebuffer([texture])

    return framebuffer, texture


def render_points_to_texture(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    colors: Optional[np.ndarray] = None,
    point_size: float = 1.0,
    width: int = 1024,
    height: int = 1024,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Render points to a texture using ModernGL.

    Args:
        x_coords: X coordinates of points
        y_coords: Y coordinates of points
        colors: Optional RGBA colors for each point (Nx4 array)
        point_size: Size of points in pixels
        width: Texture width in pixels
        height: Texture height in pixels
        x_range: (min, max) range for x coordinates
        y_range: (min, max) range for y coordinates
        background_color: Background color as (r, g, b, a) tuple

    Returns:
        RGBA texture data as numpy array with shape (height, width, 4)
    """
    ctx = ModernGLContext().get_context()

    # Create framebuffer
    framebuffer, texture = create_framebuffer(width, height)

    # Normalize coordinates to [-1, 1] range for OpenGL
    x_norm = 2.0 * (x_coords - x_range[0]) / (x_range[1] - x_range[0]) - 1.0
    y_norm = 2.0 * (y_coords - y_range[0]) / (y_range[1] - y_range[0]) - 1.0

    # Create vertex data
    n_points = len(x_coords)
    vertices = np.column_stack([x_norm, y_norm]).astype(np.float32)

    # Handle colors
    if colors is None:
        # Default to white points
        vertex_colors = np.ones((n_points, 4), dtype=np.float32)
    else:
        vertex_colors = np.asarray(colors, dtype=np.float32)
        if vertex_colors.shape[1] == 3:
            # Add alpha channel
            alpha = np.ones((n_points, 1), dtype=np.float32)
            vertex_colors = np.column_stack([vertex_colors, alpha])

    # Create vertex buffer
    vertex_data = np.column_stack([vertices, vertex_colors]).astype(np.float32)
    vbo = ctx.buffer(vertex_data.tobytes())

    # Vertex shader
    vertex_shader = (
        """
    #version 330

    in vec2 in_position;
    in vec4 in_color;

    out vec4 color;

    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        gl_PointSize = """
        + str(point_size)
        + """;
        color = in_color;
    }
    """
    )

    # Fragment shader
    fragment_shader = """
    #version 330

    in vec4 color;
    out vec4 fragColor;

    void main() {
        // Create circular points
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (length(coord) > 0.5) {
            discard;
        }
        fragColor = color;
    }
    """

    # Create program
    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    # Create vertex array
    vao = ctx.vertex_array(program, [(vbo, "2f 4f", "in_position", "in_color")])

    # Render
    framebuffer.use()
    framebuffer.clear(*background_color)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    vao.render(moderngl.POINTS)

    # Read back texture data
    texture_data = texture.read()

    # Convert to numpy array and reshape
    data = np.frombuffer(texture_data, dtype=np.uint8)
    data = data.reshape((height, width, 4))

    # Flip vertically (OpenGL uses bottom-left origin)
    data = np.flip(data, axis=0)

    # Cleanup
    vao.release()
    vbo.release()
    program.release()
    framebuffer.release()
    texture.release()

    return data


def render_triangles_to_texture(
    triangles: np.ndarray,
    colors: Optional[np.ndarray] = None,
    width: int = 1024,
    height: int = 1024,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Render filled triangles to a texture using ModernGL.

    Args:
        triangles: Triangle vertices with shape (T, 3, 2) in the same units as x_range/y_range.
        colors: Optional per-triangle colors as (T, 4) or (T, 3) array. If omitted, defaults to opaque white.
        width: Texture width in pixels.
        height: Texture height in pixels.
        x_range: (min, max) range for x coordinates.
        y_range: (min, max) range for y coordinates.
        background_color: Background color as (r, g, b, a) tuple.

    Returns:
        RGBA texture data as numpy array with shape (height, width, 4).
    """
    ctx = ModernGLContext().get_context()

    # Create framebuffer
    framebuffer, texture = create_framebuffer(width, height)

    # Validate and normalize triangle input
    tri = np.asarray(triangles, dtype=np.float32)
    if tri.ndim != 3 or tri.shape[1] != 3 or tri.shape[2] != 2:
        raise ValueError("triangles must have shape (T, 3, 2)")

    T = tri.shape[0]

    # Normalize positions to NDC [-1, 1]
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    tri_ndc = np.empty_like(tri, dtype=np.float32)
    tri_ndc[..., 0] = 2.0 * (tri[..., 0] - x0) / (x1 - x0) - 1.0
    tri_ndc[..., 1] = 2.0 * (tri[..., 1] - y0) / (y1 - y0) - 1.0

    # Colors per triangle -> per vertex
    if colors is None:
        cols = np.ones((T, 4), dtype=np.float32)
    else:
        cols = np.asarray(colors, dtype=np.float32)
        if cols.ndim == 1:
            cols = np.tile(cols[None, ...], (T, 1))
        if cols.shape[-1] == 3:
            alpha = np.ones((cols.shape[0], 1), dtype=np.float32)
            cols = np.concatenate([cols, alpha], axis=1)
        if cols.shape != (T, 4):
            raise ValueError("colors must have shape (T, 4) or (T, 3)")

    cols_per_vertex = np.repeat(cols[:, None, :], 3, axis=1)  # (T, 3, 4)

    # Interleave positions and colors per vertex
    vertex_data = np.concatenate([tri_ndc, cols_per_vertex], axis=2)  # (T, 3, 6)
    vertex_data = vertex_data.reshape((-1, 6)).astype(np.float32)
    vbo = ctx.buffer(vertex_data.tobytes())

    vertex_shader = """
    #version 330
    in vec2 in_position;
    in vec4 in_color;
    out vec4 v_color;
    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        v_color = in_color;
    }
    """

    fragment_shader = """
    #version 330
    in vec4 v_color;
    out vec4 fragColor;
    void main() {
        fragColor = v_color;
    }
    """

    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    vao = ctx.vertex_array(program, [(vbo, "2f 4f", "in_position", "in_color")])

    # Render
    framebuffer.use()
    framebuffer.clear(*background_color)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    vao.render(mode=moderngl.TRIANGLES)

    # Read back texture data
    texture_data = texture.read()
    data = np.frombuffer(texture_data, dtype=np.uint8).reshape((height, width, 4))
    data = np.flip(data, axis=0)

    # Cleanup
    vao.release()
    vbo.release()
    program.release()
    framebuffer.release()
    texture.release()

    return data


def render_lines_to_texture(
    segments: np.ndarray,
    colors: Optional[np.ndarray] = None,
    line_width: float = 1.0,
    width: int = 1024,
    height: int = 1024,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Render line segments (outlines) to a texture using triangles to achieve
    consistent pixel widths.

    Args:
        segments: Array of shape (S, 2, 2) with endpoints [[x0,y0],[x1,y1]] in data units.
        colors: Optional per-segment colors as (S,4) or (S,3). Defaults to opaque white.
        line_width: Line width in pixels.
        width: Texture width in pixels.
        height: Texture height in pixels.
        x_range: (min, max) x range of data coordinates.
        y_range: (min, max) y range of data coordinates.
        background_color: RGBA background color.

    Returns:
        RGBA numpy array with shape (height, width, 4).
    """
    ctx = ModernGLContext().get_context()

    segs = np.asarray(segments, dtype=np.float32)
    if segs.ndim != 3 or segs.shape[1] != 2 or segs.shape[2] != 2:
        raise ValueError("segments must have shape (S, 2, 2)")

    S = segs.shape[0]
    if S == 0:
        # Return empty transparent image
        framebuffer, texture = create_framebuffer(width, height)
        framebuffer.use()
        framebuffer.clear(*background_color)
        data = np.frombuffer(texture.read(), dtype=np.uint8).reshape((height, width, 4))
        data = np.flip(data, axis=0)
        framebuffer.release()
        texture.release()
        return data

    # Map data coords -> pixel coords
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])

    # Avoid division by zero
    xr = x1 - x0 if (x1 - x0) != 0 else 1.0
    yr = y1 - y0 if (y1 - y0) != 0 else 1.0

    p0 = segs[:, 0, :].astype(np.float32)
    p1 = segs[:, 1, :].astype(np.float32)

    # Pixel coordinates
    p0_pix = np.empty_like(p0, dtype=np.float32)
    p1_pix = np.empty_like(p1, dtype=np.float32)
    p0_pix[:, 0] = (p0[:, 0] - x0) / xr * float(width)
    p0_pix[:, 1] = (p0[:, 1] - y0) / yr * float(height)
    p1_pix[:, 0] = (p1[:, 0] - x0) / xr * float(width)
    p1_pix[:, 1] = (p1[:, 1] - y0) / yr * float(height)

    # Build quad per segment in pixel space with half-width
    hw = float(line_width) * 0.5
    quads = []
    valid_idx = []

    for i in range(S):
        a = p0_pix[i]
        b = p1_pix[i]
        d = b - a
        norm = np.linalg.norm(d)
        if norm <= 1e-6:
            continue  # skip degenerate segments

        # Perpendicular of (dx,dy) in pixel space
        n = np.array([-d[1], d[0]], dtype=np.float32)
        n_norm = np.linalg.norm(n)
        if n_norm <= 1e-12:
            continue
        n = (n / n_norm) * hw

        # Quad vertices in pixel coords: a+n, a-n, b-n, b+n (clockwise)
        v0 = a + n
        v1 = a - n
        v2 = b - n
        v3 = b + n

        # Convert to NDC
        def to_ndc(pix: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    2.0 * (pix[0] / float(width)) - 1.0,
                    2.0 * (pix[1] / float(height)) - 1.0,
                ],
                dtype=np.float32,
            )

        v0n = to_ndc(v0)
        v1n = to_ndc(v1)
        v2n = to_ndc(v2)
        v3n = to_ndc(v3)

        # Two triangles: (v0,v1,v2) and (v0,v2,v3)
        quads.append(np.stack([v0n, v1n, v2n, v0n, v2n, v3n], axis=0))
        valid_idx.append(i)

    if len(quads) == 0:
        framebuffer, texture = create_framebuffer(width, height)
        framebuffer.use()
        framebuffer.clear(*background_color)
        data = np.frombuffer(texture.read(), dtype=np.uint8).reshape((height, width, 4))
        data = np.flip(data, axis=0)
        framebuffer.release()
        texture.release()
        return data

    verts_ndc = np.concatenate(quads, axis=0).astype(np.float32)  # (N*6, 2)

    # Colors
    if colors is None:
        cols = np.ones((len(valid_idx), 4), dtype=np.float32)
    else:
        cols = np.asarray(colors, dtype=np.float32)
        if cols.ndim == 1:
            cols = np.tile(cols[None, ...], (S, 1))
        if cols.shape[-1] == 3:
            alpha = np.ones((cols.shape[0], 1), dtype=np.float32)
            cols = np.concatenate([cols, alpha], axis=1)
        if cols.shape[0] != S:
            raise ValueError("colors must provide one color per segment")
        cols = cols[valid_idx, :].astype(np.float32)

    # Expand per-quad color to per-vertex (6 vertices per segment)
    cols_per_vertex = (
        np.repeat(cols[:, None, :], 6, axis=1).reshape((-1, 4)).astype(np.float32)
    )

    # Interleave position and color
    vertex_data = np.column_stack([verts_ndc, cols_per_vertex]).astype(np.float32)

    # Create GL resources
    framebuffer, texture = create_framebuffer(width, height)
    vbo = ctx.buffer(vertex_data.tobytes())

    vertex_shader = """
    #version 330
    in vec2 in_position;
    in vec4 in_color;
    out vec4 v_color;
    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        v_color = in_color;
    }
    """

    fragment_shader = """
    #version 330
    in vec4 v_color;
    out vec4 fragColor;
    void main() {
        fragColor = v_color;
    }
    """

    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    vao = ctx.vertex_array(program, [(vbo, "2f 4f", "in_position", "in_color")])

    # Render
    framebuffer.use()
    framebuffer.clear(*background_color)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    vao.render(mode=moderngl.TRIANGLES)

    # Readback
    data = np.frombuffer(texture.read(), dtype=np.uint8).reshape((height, width, 4))
    data = np.flip(data, axis=0)

    # Cleanup
    vao.release()
    vbo.release()
    program.release()
    framebuffer.release()
    texture.release()

    return data


def render_rectangles_to_texture(
    x: np.ndarray,
    y: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    colors: Optional[np.ndarray] = None,
    width: int = 1024,
    height: int = 1024,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Render filled rectangles to a texture using ModernGL.

    Args:
        x: X positions of rectangle left edges
        y: Y positions of rectangle bottom edges
        widths: Widths of rectangles
        heights: Heights of rectangles
        colors: Optional per-rectangle colors as (R, 4) or (R, 3) array
        width: Texture width in pixels
        height: Texture height in pixels
        x_range: (min, max) range for x coordinates
        y_range: (min, max) range for y coordinates
        background_color: Background color as (r, g, b, a) tuple

    Returns:
        RGBA texture data as numpy array with shape (height, width, 4)
    """
    ctx = ModernGLContext().get_context()

    # Validate inputs
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)
    heights = np.asarray(heights, dtype=np.float32)

    if not (x.shape == y.shape == widths.shape == heights.shape):
        raise ValueError("x, y, widths, and heights must have the same shape")

    R = len(x)
    if R == 0:
        # Return empty transparent image
        framebuffer, texture = create_framebuffer(width, height)
        framebuffer.use()
        framebuffer.clear(*background_color)
        data = np.frombuffer(texture.read(), dtype=np.uint8).reshape((height, width, 4))
        data = np.flip(data, axis=0)
        framebuffer.release()
        texture.release()
        return data

    # Normalize coordinates to NDC [-1, 1]
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])

    # Avoid division by zero
    xr = x1 - x0 if (x1 - x0) != 0 else 1.0
    yr = y1 - y0 if (y1 - y0) != 0 else 1.0

    # Calculate rectangle corners in NDC
    # Each rectangle: (x, y) is bottom-left, width/height extend right/up
    x_left = 2.0 * (x - x0) / xr - 1.0
    x_right = 2.0 * (x + widths - x0) / xr - 1.0
    y_bottom = 2.0 * (y - y0) / yr - 1.0
    y_top = 2.0 * (y + heights - y0) / yr - 1.0

    # Build triangles for each rectangle (2 triangles per rect = 6 vertices)
    # Triangle 1: bottom-left, bottom-right, top-right
    # Triangle 2: bottom-left, top-right, top-left
    verts = []
    for i in range(R):
        v0 = [x_left[i], y_bottom[i]]  # bottom-left
        v1 = [x_right[i], y_bottom[i]]  # bottom-right
        v2 = [x_right[i], y_top[i]]  # top-right
        v3 = [x_left[i], y_top[i]]  # top-left

        # Two triangles
        verts.append([v0, v1, v2])  # tri 1
        verts.append([v0, v2, v3])  # tri 2

    verts_ndc = np.concatenate(verts, axis=0).astype(np.float32)  # (R*6, 2)

    # Handle colors
    if colors is None:
        cols = np.ones((R, 4), dtype=np.float32)
    else:
        cols = np.asarray(colors, dtype=np.float32)
        if cols.ndim == 1:
            cols = np.tile(cols[None, ...], (R, 1))
        if cols.shape[-1] == 3:
            alpha = np.ones((cols.shape[0], 1), dtype=np.float32)
            cols = np.concatenate([cols, alpha], axis=1)
        if cols.shape[0] != R:
            raise ValueError("colors must provide one color per rectangle")
        cols = cols.astype(np.float32)

    # Expand per-rectangle color to per-vertex (6 vertices per rectangle)
    cols_per_vertex = (
        np.repeat(cols[:, None, :], 6, axis=1).reshape((-1, 4)).astype(np.float32)
    )

    # Interleave position and color
    vertex_data = np.column_stack([verts_ndc, cols_per_vertex]).astype(np.float32)

    # Create framebuffer and GL resources
    framebuffer, texture = create_framebuffer(width, height)
    vbo = ctx.buffer(vertex_data.tobytes())

    vertex_shader = """
    #version 330
    in vec2 in_position;
    in vec4 in_color;
    out vec4 v_color;
    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        v_color = in_color;
    }
    """

    fragment_shader = """
    #version 330
    in vec4 v_color;
    out vec4 fragColor;
    void main() {
        fragColor = v_color;
    }
    """

    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    vao = ctx.vertex_array(program, [(vbo, "2f 4f", "in_position", "in_color")])

    # Render
    framebuffer.use()
    framebuffer.clear(*background_color)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    vao.render(mode=moderngl.TRIANGLES)

    # Read back texture data
    texture_data = texture.read()
    data = np.frombuffer(texture_data, dtype=np.uint8).reshape((height, width, 4))
    data = np.flip(data, axis=0)

    # Cleanup
    vao.release()
    vbo.release()
    program.release()
    framebuffer.release()
    texture.release()

    return data


def calculate_dpi_size(
    width_mm: float, height_mm: float, dpi: float
) -> Tuple[int, int]:
    """
    Calculate pixel dimensions from physical dimensions and DPI.

    Args:
        width_mm: Width in millimeters
        height_mm: Height in millimeters
        dpi: Dots per inch

    Returns:
        Tuple of (width_pixels, height_pixels)
    """
    # Convert mm to inches
    width_inches = width_mm / 25.4
    height_inches = height_mm / 25.4

    # Calculate pixel dimensions
    width_pixels = int(width_inches * dpi)
    height_pixels = int(height_inches * dpi)

    return width_pixels, height_pixels
