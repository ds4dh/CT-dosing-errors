from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple, Literal

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
    ext: str  # original file extension if preserved, else "png"


# ---------- Text extraction ----------

def extract_text_from_pdf(
        pdf_path: str,
        *,
        mode: Literal["simple", "layout"] = "simple",
        join_with: str = "\n\n",
) -> str:
    """
    Fast text extraction using PyMuPDF (MuPDF).

    mode="simple": page.get_text("text")  -> plain reading order text.
    mode="layout": page.get_text("blocks") -> block-wise; we order blocks top-left to bottom-right.
    """
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            if mode == "simple":
                pages.append(page.get_text("text") or "")
            else:
                # blocks: list of (x0, y0, x1, y1, "text", block_no, block_type, ...)
                blocks = page.get_text("blocks") or []
                blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))  # sort by y, then x
                pages.append("\n".join(b[4].rstrip() for b in blocks if b[4]))
    return join_with.join(pages)


class DeepSeekOCRExtractor:
    """
    DeepSeek-OCR wrapper.

    - OCR logic:
      * Render pages via pypdfium2 at DPI -> PIL RGB
      * Prompt: "<image>\\n<|grounding|>Convert the document to markdown."
      * vLLM.generate with multi_modal_data={"image": PIL.Image}
      * SamplingParams: temperature=0.0, max_tokens=8192
        extra_args={'ngram_size':30, 'window_size':90, 'whitelist_token_ids':{128821,128822}}
      * skip_special_tokens=False
      * logits_processors=[NGramPerReqLogitsProcessor]

    Usage:
        ocr = DeepSeekOCRExtractor()
        text = ocr.extract_text_from_pdf("/path/file.pdf")            # joined with "\n\n"
        text = ocr.extract_text_from_pdf("/path/file.pdf", join_with="")  # no separator
    """

    PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

    def __init__(
            self,
            *,
            model_name: str = "deepseek-ai/DeepSeek-OCR",
            temperature: float = 0.0,
            max_tokens: int = 8192,
            dpi: int = 300,
    ) -> None:

        import pypdfium2 as pdfium
        from vllm import LLM, SamplingParams
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        self._pdfium = pdfium
        self._dpi = dpi

        self._llm = LLM(
            model=model_name,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
        )
        self._sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # <td>, </td>
            ),
            skip_special_tokens=False,
        )

    def extract_text_from_pdf(
            self,
            pdf_path: str,
            *,
            join_with: str = "\n\n",
    ) -> str:
        """
        OCR the entire PDF and return a single markdown string joined with `join_with`.
        """
        pdfium = self._pdfium
        scale = self._dpi / 72.0

        doc = pdfium.PdfDocument(pdf_path)
        try:
            items: List[Image.Image] = []
            for page in doc:
                pil_img = page.render(scale=scale).to_pil().convert("RGB")
                items.append(pil_img)
        finally:
            doc.close()

        model_inputs = [
            {"prompt": self.PROMPT, "multi_modal_data": {"image": img}}
            for img in items
        ]
        outputs = self._llm.generate(model_inputs, self._sampling_params)

        page_texts = [
            output.outputs[0].text if getattr(output, "outputs", None) else ""
            for output in outputs
        ]

        return join_with.join(page_texts)


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
                        info = doc.extract_image(
                            xref)  # {'image': bytes, 'width': int, 'height': int, 'ext': 'jpeg', ...}
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
