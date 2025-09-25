import moderngl
import numpy as np
from typing import Optional, Tuple
import threading


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
            self.ctx = moderngl.create_context(standalone=True, require=330)
            self._initialized = True

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
