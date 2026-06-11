"""
downscale.py
------------
Average-pooling downscaler for square, single-channel (grayscale) images.
Written for MicroPython on the ESP32; the @micropython.native decorator
compiles the hot loops to native machine code for a meaningful speed-up.

Usage
-----
    import downscale
    downscale.downscale(src_buf, dst_buf, in_size=96, out_size=32)

Both buffers must be flat array.array('B', ...) objects.
in_size must be an exact multiple of out_size.
"""

import array
import time


@micropython.native
def _average2d(inp, rowstride: int, x: int, y: int, size: int) -> int:
    """Return the integer average of a (size × size) block starting at (x, y)."""
    acc: int = 0
    for r in range(y, y + size):
        for c in range(x, x + size):
            acc += inp[(r * rowstride) + c]
    return acc // (size * size)


@micropython.native
def downscale(inp, out, in_size: int, out_size: int) -> None:
    """
    Average-pool a square grayscale image from in_size×in_size
    down to out_size×out_size.

    Parameters
    ----------
    inp     : flat buffer of length in_size²  (read-only)
    out     : flat buffer of length out_size² (written in-place)
    in_size : side length of the input image
    out_size: side length of the output image
    """
    assert len(inp) == in_size * in_size
    assert len(out) == out_size * out_size
    assert (in_size % out_size) == 0, (in_size, out_size)

    factor: int = in_size // out_size

    for row in range(out_size):
        for col in range(out_size):
            out[(row * out_size) + col] = _average2d(
                inp, in_size, col * factor, row * factor, factor
            )


# ---------------------------------------------------------------------------
# Self-test (requires npyfile module and inp.npy on the filesystem)
# ---------------------------------------------------------------------------

def _test() -> None:
    import npyfile

    shape, data = npyfile.load("inp.npy")
    npyfile.save("orig.npy", data, shape)

    in_size, out_size = 96, 32
    out = array.array("B", (0 for _ in range(out_size * out_size)))

    t0 = time.ticks_ms()
    downscale(data, out, in_size, out_size)
    print(f"downscale: {time.ticks_diff(time.ticks_ms(), t0)} ms")

    npyfile.save("out.npy", out, (out_size, out_size))


if __name__ == "__main__":
    _test()
