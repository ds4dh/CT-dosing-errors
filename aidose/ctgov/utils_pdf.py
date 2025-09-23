from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple, Literal

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image


# ---------- Data models ----------

@dataclass(frozen=True)
class ExtractedImage:
    page_number: int
    xref: int
    image: Image.Image
    width: int
    height: int
    colorspace: str  # e.g., "RGB", "RGBA", "L", "LA"
    ext: str         # original file extension if preserved, else "png"


# ---------- Text extraction ----------

def extract_text_from_pdf(
    pdf_path: str,
    *,
    mode: Literal["simple", "layout"] = "simple",
    join_with: str = "\n\n",
) -> str:
    """
    Extract text from a PDF.

    :param pdf_path: Path to the PDF file.
    :param mode: 'simple' uses pdfplumber's default. 'layout' stitches words by line to better
                 preserve columns (heuristic).
    :param join_with: Separator placed between pages in the final string.
    :return: A single string containing extracted text from all pages.
    """
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if mode == "simple":
                pages.append(page.extract_text() or "")
            else:
                # Layout-aware: group words into lines by their top coordinate
                words = page.extract_words(use_text_flow=True) or []
                lines_by_y: dict[int, list[tuple[float, str]]] = {}
                for w in words:
                    y_key = int(round(w["top"]))
                    lines_by_y.setdefault(y_key, []).append((w["x0"], w["text"]))
                ordered_lines = []
                for y in sorted(lines_by_y):
                    line = " ".join(text for _, text in sorted(lines_by_y[y], key=lambda p: p[0]))
                    ordered_lines.append(line)
                pages.append("\n".join(ordered_lines))
    return join_with.join(pages)


# ---------- Image extraction ----------

def extract_images_from_pdf(
    pdf_path: str,
    *,
    dedupe: bool = True,
    keep_original_format: bool = True,
    min_dimensions: Tuple[int, int] = (64, 64),
) -> List[ExtractedImage]:
    """
    Extract embedded raster images from a PDF and return them as PIL Images (in-memory).

    Strategy:
      - Prefer extracting the original image stream (JPEG/JP2/PNG/JBIG2/etc.) via PyMuPDF.
      - If extraction or decoding fails, fall back to rendering the XObject into an RGB(A) pixmap.
      - De-duplicate by XREF if requested.

    :param pdf_path: Path to the PDF file.
    :param dedupe: Skip images with XREFs already seen.
    :param keep_original_format: Preserve the original encoded streams when possible.
    :param min_dimensions: Skip images smaller than (width, height).
    :return: List of ExtractedImage objects with PIL images and metadata.
    """
    # TODO: Exclude all-black/white images or those with very low variance.
    results: List[ExtractedImage] = []
    seen_xrefs: set[int] = set()

    def _render_pixmap(doc: fitz.Document, xref: int, page_number: int) -> ExtractedImage | None:
        # Create Pixmap and normalize to RGB(A) if needed
        pix = fitz.Pixmap(doc, xref)
        if pix.n >= 5:  # e.g., CMYK/Indexed -> convert to RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)

        w, h = pix.width, pix.height
        if w < min_dimensions[0] or h < min_dimensions[1]:
            return None

        # Map Pixmap channels to PIL mode
        if pix.n == 1:
            mode = "L"
        elif pix.n == 2:
            mode = "LA"
        elif pix.n == 3:
            mode = "RGB"
        elif pix.n == 4:
            mode = "RGBA"
        else:
            # Fallback: convert to RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
            mode = "RGB"

        img = Image.frombytes(mode, (w, h), pix.samples)
        return ExtractedImage(
            page_number=page_number,
            xref=xref,
            image=img,
            width=w,
            height=h,
            colorspace=mode,
            ext="png",  # rendered images are effectively PNG-equivalent in-memory
        )

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                width, height = img_info[2], img_info[3]

                if width < min_dimensions[0] or height < min_dimensions[1]:
                    continue
                if dedupe and xref in seen_xrefs:
                    continue

                added = False
                if keep_original_format:
                    try:
                        info = doc.extract_image(xref)  # {'image': bytes, 'width': int, 'height': int, 'ext': 'jpeg', ...}
                        if info and info.get("image"):
                            w, h = info["width"], info["height"]
                            if w >= min_dimensions[0] and h >= min_dimensions[1]:
                                # Try to decode with PIL; if it fails (e.g., JBIG2), render instead
                                try:
                                    img = Image.open(BytesIO(info["image"]))
                                    img.load()  # ensure data is read now (not lazy)
                                    results.append(
                                        ExtractedImage(
                                            page_number=page_index + 1,
                                            xref=xref,
                                            image=img,
                                            width=w,
                                            height=h,
                                            colorspace=img.mode,
                                            ext=info.get("ext", "bin"),
                                        )
                                    )
                                    added = True
                                except Exception:
                                    pass
                    except Exception:
                        # ignore and fallback to render
                        pass

                if not added:
                    rendered = _render_pixmap(doc, xref, page_index + 1)
                    if rendered:
                        results.append(rendered)

                if dedupe:
                    seen_xrefs.add(xref)

    return results
