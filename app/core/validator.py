"""
Image Validation Module for ORBIT.

Provides deep verification of uploaded imagery before entering the OpenCV/ML pipeline:
1. Filename & Extension sanitization
2. Header & Magic Byte verification using PIL (prevents spoofed or corrupt files)
3. Dimension constraint checks (guards against memory exhaustion / min size thresholds)
4. Stream pointer management (ensures downstream functions can re-read the file)
"""

import os
from PIL import Image
from werkzeug.datastructures import FileStorage

def get_file_extension(filename: str) -> str:
    """Extract lowercase file extension without dot."""
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()

def validate_image_file(
    file: FileStorage,
    allowed_extensions: set = None,
    min_dim: int = 256,
    max_dim: int = 4096,
) -> tuple[bool, str | None, dict | None]:
    """
    Validates an uploaded image file storage object.

    Args:
        file: Werkzeug FileStorage object from request.files.
        allowed_extensions: Set of allowed lowercase extensions (e.g. {'png', 'jpg'}).
        min_dim: Minimum allowed width or height in pixels.
        max_dim: Maximum allowed width or height in pixels.

    Returns:
        tuple (is_valid: bool, error_message: str | None, metadata: dict | None)
    """
    if allowed_extensions is None:
        allowed_extensions = {"png", "jpg", "jpeg", "tif", "tiff"}

    if not file or not file.filename:
        return False, "No file provided or empty filename.", None

    # 1. Extension check
    ext = get_file_extension(file.filename)
    if ext not in allowed_extensions:
        return (
            False,
            f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(allowed_extensions))}.",
            None,
        )

    # 2. File size check (calculate from stream)
    file.seek(0, os.SEEK_END)
    file_size_bytes = file.tell()
    file.seek(0)  # Reset pointer to start

    if file_size_bytes == 0:
        return False, "Uploaded file is empty (0 bytes).", None

    # 3. Header & structural integrity check using PIL
    try:
        # Image.open loads metadata only, does not decompress entire pixel raster into RAM
        img = Image.open(file.stream)
        width, height = img.size
        img_format = img.format

        # Verify image integrity (catches truncated/corrupted files)
        img.verify()
    except Exception as e:
        # Reset stream pointer on failure
        file.seek(0)
        return False, f"Corrupted image or invalid header: {str(e)}", None

    # 4. Dimension constraints check
    if width < min_dim or height < min_dim:
        file.seek(0)
        return (
            False,
            f"Image dimensions ({width}x{height}) are too small. Minimum resolution is {min_dim}x{min_dim}.",
            None,
        )

    if width > max_dim or height > max_dim:
        file.seek(0)
        return (
            False,
            f"Image dimensions ({width}x{height}) exceed maximum allowed resolution ({max_dim}x{max_dim}).",
            None,
        )

    # Reset stream pointer so downstream readers (e.g. cv2 or file.save) start from byte 0
    file.seek(0)

    metadata = {
        "filename": file.filename,
        "format": img_format,
        "width": width,
        "height": height,
        "size_bytes": file_size_bytes,
        "size_kb": round(file_size_bytes / 1024, 2),
    }

    return True, None, metadata

def validate_image_pair(
    file_before: FileStorage,
    file_after: FileStorage,
    allowed_extensions: set = None,
    min_dim: int = 256,
    max_dim: int = 4096,
) -> tuple[bool, dict[str, str], dict[str, dict]]:
    """
    Validates a pair of temporal satellite images (T-0 Reference & T-1 Target).

    Returns:
        tuple (is_valid: bool, errors: dict[str, str], metadata: dict[str, dict])
    """
    errors = {}
    metadata = {}

    # Validate Before (T-0)
    ok_b, err_b, meta_b = validate_image_file(file_before, allowed_extensions, min_dim, max_dim)
    if not ok_b:
        errors["image_before"] = err_b
    else:
        metadata["image_before"] = meta_b

    # Validate After (T-1)
    ok_a, err_a, meta_a = validate_image_file(file_after, allowed_extensions, min_dim, max_dim)
    if not ok_a:
        errors["image_after"] = err_a
    else:
        metadata["image_after"] = meta_a

    return (len(errors) == 0, errors, metadata)
