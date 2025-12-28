#!/usr/bin/env python3
"""
Generate educational presentations from table of contents JSON files using Gemini on Vertex AI.

For each JSON file in toc_openai_filtered/:
1. Extracts topics and subtopics from the filtered TOC
2. Sends requests to Gemini to generate presentation text (fact-based only)
3. Uses Gemini on Vertex AI to generate slide images directly from presentation text
4. Converts slide images to PDF
5. Saves generated presentations to generated_presentations/{subject}/{form}/
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Pre-run filename shortening
# -----------------------------
def _shorten_toc_filenames_if_needed() -> None:
    """Shorten overly long filenames inside toc_openai_filtered/ before processing.

    This prevents filesystem errors like "File name too long" that may appear
    when creating directories based on long book titles. The operation is
    idempotent: if no renames are needed, it does nothing.

    Controlled by env var SHORTEN_TOC (default: enabled). Set SHORTEN_TOC=0 to disable.
    """
    try:
        if os.getenv("SHORTEN_TOC", "1").strip() in {"0", "false", "False", "no", "NO"}:
            print("[shorten] Skipped (SHORTEN_TOC disabled)")
            return

        # Import locally to avoid import cost unless needed by this script
        try:
            from tools.shorten_filenames import plan_renames, apply_renames, write_mapping  # type: ignore
        except Exception:
            # Also support alternate location if the tool was placed at project root
            try:
                # type: ignore
                from shorten_filenames import plan_renames, apply_renames, write_mapping  # type: ignore
            except Exception as e:
                print(f"[shorten] WARN: Could not import shortening utility: {e}")
                return

        root = Path("toc_openai_filtered")
        if not root.exists():
            print(f"[shorten] Root not found, skipping: {root}")
            return

        renames = plan_renames(root)
        if not renames:
            print("[shorten] No renames needed.")
            return

        applied = apply_renames(renames)
        print(f"[shorten] Applied renames: {len(applied)}/{len(renames)}")
        if applied:
            mapping_path = write_mapping(applied, Path.cwd())
            print(f"[shorten] Mapping written to: {mapping_path}")
    except Exception as e:
        # Do not fail the whole generation because of shortening errors
        print(f"[shorten] WARN: Shortening step failed: {e}")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
TOC_ROOT = PROJECT_ROOT / "toc_openai_filtered"
OUTPUT_ROOT = PROJECT_ROOT / "generated_presentations"

MAX_API_RETRIES = 3
API_RETRY_BASE_DELAY = 3.0  # seconds
GEMINI_MODEL = os.environ.get("VERTEX_TEXT_MODEL", "gemini-2.0-flash")
VERTEX_IMAGE_MODEL = os.environ.get("VERTEX_IMAGE_MODEL", "gemini-3-pro-image-preview")
NUM_WORKERS = int(os.environ.get("PRESENTATION_WORKERS", "3"))  # Number of parallel workers

# Remote image generation size (server-side). Smaller sizes are faster and cheaper.
# Allowed values: "SD", "HD", "2K" (case-insensitive). Default: "SD" for faster generation.
_REQ_IMG_SIZE_RAW = os.environ.get("PRESENTATION_IMAGE_SIZE", "HD").strip()
_REQ_IMG_SIZE = _REQ_IMG_SIZE_RAW.upper()
IMAGE_REQUEST_SIZE = _REQ_IMG_SIZE if _REQ_IMG_SIZE in {"SD", "HD", "2K"} else "SD"

# Image size/quality controls for generated presentation images
# Lower values reduce file size substantially while keeping slides readable
# Defaults tightened for faster generation pipeline and smaller outputs
IMAGE_MAX_WIDTH = int(os.environ.get("PRESENTATION_IMAGE_MAX_WIDTH", "1024"))
IMAGE_JPEG_QUALITY = int(os.environ.get("PRESENTATION_IMAGE_QUALITY", "70"))

# Thread lock for safe printing
print_lock = threading.Lock()


def _load_env_file(env_path: Path) -> Dict[str, str]:
    """Very small .env reader (no external deps)."""
    env: Dict[str, str] = {}
    try:
        if not env_path.exists():
            return env
        content = env_path.read_text(encoding="utf-8")
    except OSError:
        return env
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env[key] = value
    return env


def _load_vertex_env_defaults() -> Tuple[str, str]:
    """Load project/region from .env if not present in the environment."""
    env_path = (PROJECT_ROOT / ".env").resolve()
    env_map = _load_env_file(env_path)
    for key in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_REGION"):
        if key not in os.environ and key in env_map:
            os.environ[key] = env_map[key]
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "long-grin-481614-s3")
    region = os.environ.get("GOOGLE_CLOUD_REGION", "global")
    return project, region


def _init_vertex_genai_client() -> Tuple[Any, Any]:
    """Initialize google-genai client for Vertex AI."""
    project, region = _load_vertex_env_defaults()
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Required library not installed: {e}. Install with: pip install google-genai"
        )
    client = genai.Client(vertexai=True, project=project, location=region)
    return client, types


def generate_presentation_with_gemini(
    presentation_text: str,
    topic_title: str,
    subject: str,
    form: int,
) -> Tuple[bool, Optional[bytes]]:
    """
    Generate graphical PDF presentation using Gemini on Vertex AI.
    
    Uses Gemini on Vertex AI to generate graphical presentation slides with images and visual elements,
    then converts to PDF.
    
    Args:
        presentation_text: The presentation content text generated by Gemini
        topic_title: Title of the topic
        subject: Subject name
        form: Form/grade number
    
    Returns:
        (ok, pdf_bytes_or_error)
    """
    try:
        import io
        import tempfile
        from pathlib import Path as PathLib
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from PIL import Image as PILImage
        
        def _compress_image_to_jpeg(src_path: PathLib, dst_path: PathLib,
                                    max_width: int = IMAGE_MAX_WIDTH,
                                    quality: int = IMAGE_JPEG_QUALITY) -> PathLib:
            try:
                img = PILImage.open(str(src_path))
                # Convert to RGB to ensure JPEG compatibility
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                # Downscale preserving aspect ratio if wider than max_width
                w, h = img.size
                if w > max_width:
                    new_w = max_width
                    new_h = int(h * (new_w / float(w)))
                    img = img.resize((new_w, new_h), PILImage.LANCZOS)
                # Save as optimized JPEG
                img.save(
                    str(dst_path),
                    format="JPEG",
                    quality=max(1, min(95, int(quality))),
                    optimize=True,
                    progressive=True,
                    subsampling=1,  # better quality/size balance
                )
                return dst_path
            except Exception:
                # Fallback: if compression fails, just return original path
                return src_path
        
        # Step 1: Parse presentation text into logical slides
        # Determine language based on subject
        is_english_subject = subject.lower() in ["english", "англійська", "англійська_мова"]
        presentation_language = "English" if is_english_subject else "Ukrainian"
        
        # Split presentation text into logical slides (by paragraphs or sections)
        # Simple approach: split by double newlines or numbered sections
        slides = []
        paragraphs = [p.strip() for p in presentation_text.split('\n\n') if p.strip()]
        
        # Group paragraphs into slides (3-4 paragraphs per slide, or split by numbered sections)
        current_slide_content = []
        slide_num = 1
        
        for para in paragraphs:
            # Check if this looks like a new section/topic (starts with number, bold text, etc.)
            is_new_section = (
                re.match(r'^\d+[\.\)]\s+', para) or
                re.match(r'^[А-ЯІЇЄҐ]', para) or  # Starts with capital Ukrainian letter
                len(para) < 100  # Short paragraph might be a title
            )
            
            if is_new_section and current_slide_content and len(current_slide_content) >= 2:
                # Create slide from accumulated content
                slide_title = current_slide_content[0][:100] if current_slide_content else topic_title
                slide_content = current_slide_content[1:4] if len(current_slide_content) > 1 else current_slide_content
                slides.append({
                    "title": slide_title,
                    "content": slide_content[:4],  # Max 4 items, already a list
                    "image_prompt": ""
                })
                current_slide_content = [para]
                slide_num += 1
            else:
                current_slide_content.append(para)
            
            # Limit slides to reasonable number
            if slide_num > 10:
                break
        
        # Add remaining content as last slide
        if current_slide_content:
            slide_title = current_slide_content[0][:100] if current_slide_content else topic_title
            slide_content = current_slide_content[1:4] if len(current_slide_content) > 1 else current_slide_content
            slides.append({
                "title": slide_title,
                "content": slide_content[:4],  # Already a list
                "image_prompt": ""
            })

        # Ensure we have at least one slide
        if not slides:
            # Fallback: create one slide with the entire text
            slides = [{
                "title": topic_title,
                "content": [presentation_text[:500]],
                "image_prompt": ""
            }]

        # Remove accidental duplicate cover/title slide from content slides
        topic_norm = re.sub(r"\s+", " ", topic_title.strip().lower())
        subject_norm = re.sub(r"\s+", " ", subject.strip().lower())
        form_token = str(form).strip()
        cleaned_slides = []
        for s in slides:
            title_raw = str(s.get("title", "")).strip()
            title_norm = re.sub(r"\s+", " ", title_raw.lower())
            content_list = [c for c in s.get("content", []) if isinstance(c, str)]
            content_join = " ".join(content_list).lower()
            has_class = bool(re.search(r"\b(клас|class|grade)\b", content_join))
            has_form_num = form_token and form_token in content_join
            has_subject = subject_norm and subject_norm in content_join
            content_items = [c for c in content_list if c.strip()]
            # If a slide looks like a cover, skip it
            if title_norm == topic_norm and len(content_items) <= 3:
                continue
            if len(content_items) <= 3 and (has_class or has_form_num or has_subject):
                continue
            cleaned_slides.append(s)
        slides = cleaned_slides
        
        # Step 2: Generate complete slide images using Gemini on Vertex AI
        try:
            client, types = _init_vertex_genai_client()
            model_id = VERTEX_IMAGE_MODEL
            with print_lock:
                print(f"    ✓ Using Vertex AI Gemini Image model: {model_id}")
        except Exception as e:
            return False, f"Failed to initialize Vertex Gemini client: {e}"
        
        # Generate title slide image
        temp_dir = PathLib(tempfile.mkdtemp())
        generated_images = []
        
        # Get subject-based theme - STRICTLY according to subject
        def get_subject_theme(subject_name: str) -> str:
            """Get theme description based on subject - STRICTLY enforced."""
            themes = {
                "ukrainian_language": "Ukrainian language and literature theme with books, letters, Ukrainian alphabet, and Ukrainian cultural elements. Blue and yellow color scheme (Ukrainian flag colors).",
                "ukrainian_literature": "Ukrainian literary theme with books, quills, classic Ukrainian literature elements, Ukrainian writers. Warm, elegant colors with cultural motifs.",
                "mathematics": "Mathematical theme with geometric shapes, formulas, numbers, equations, graphs. Clean, modern design with blue and white. NO other subjects mixed in.",
                "algebra": "Algebraic theme with equations, graphs, mathematical symbols, algebraic expressions. Professional blue and gray color scheme. STRICTLY mathematical.",
                "geometry": "Geometric theme with shapes, angles, patterns, geometric figures. Bright, clear colors with geometric designs. STRICTLY geometric.",
                "biology": "Biological theme with nature, plants, animals, cells, biological processes, ecosystems. Green and natural color palette. STRICTLY biological.",
                "chemistry": "Chemical theme with molecules, lab equipment, periodic table, chemical reactions, test tubes. Scientific blue and white theme. STRICTLY chemical.",
                "physics": "Physics theme with atoms, forces, energy concepts, physical laws, laboratory equipment. Modern blue and purple color scheme. STRICTLY physical.",
                "history": "Historical theme with maps, documents, historical artifacts, Ukrainian history elements. Classic, scholarly brown and gold tones. STRICTLY historical.",
                "world_history": "World history theme with globes, historical events, civilizations, world maps. Rich, historical color palette. STRICTLY world history.",
                "geography": "Geographic theme with maps, landscapes, earth elements, geographic features. Natural green and blue colors. STRICTLY geographic.",
                "informatics": "Technology theme with computers, code, digital elements, programming concepts. Modern tech blue and black theme. STRICTLY technological.",
                "art": "Artistic theme with colors, brushes, creative elements, art supplies, paintings. Vibrant, artistic color palette. STRICTLY artistic.",
                "english": "English language learning theme with English books, letters, language learning elements, UK/US cultural elements. Red, white, and blue color scheme.",
            }
            # Normalize subject name
            subject_lower = subject_name.lower().replace(" ", "_")
            return themes.get(subject_lower, f"Professional educational theme for {subject_name} with clean, modern design. Blue and white color scheme.")
        
        subject_theme = get_subject_theme(subject)
        
        # Determine language for text rendering
        is_english_subject = subject.lower() in ["english", "англійська", "англійська_мова"]
        text_language = "English" if is_english_subject else "Ukrainian"
        temp_dir = PathLib(tempfile.mkdtemp())
        generated_images = []
        
        # Generate title slide using Gemini 3 Image Preview
        # STRICTLY enforce subject theme and language
        title_slide_prompt = (
            f"Create a high-quality educational presentation cover slide. "
            f"VISUALS: {subject_theme} STRICTLY follow this theme - do not mix themes from other subjects. "
            f"LAYOUT: Modern, clean, readable font, high contrast. "
            f"TEXT: Render ONLY the topic title '{topic_title}' in {text_language}. "
            f"Do NOT add grade/class number. Do NOT add words like Slide/Слайд or any numbering. "
            f"Do NOT add any UI labels like 'header', 'headline', 'title', 'body'. "
            f"Do NOT render the words 'title slide' or any equivalent. "
            f"Do NOT use markdown or list markers (#, ##, *, -, 1.). "
            f"Do NOT render JSON keys or labels like name/subject. "
            f"All visible text must be {text_language} and legible on the background. "
            f"Theme MUST be strictly {subject} themed - no mixing with other subjects."
        )
        
        try:
            with print_lock:
                print(f"      🎨 Generating title slide...")
            
            response = client.models.generate_content(
                model=model_id,
                contents=title_slide_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        aspect_ratio="16:9",
                        image_size="2K",
                    ),
                ),
            )
            
            # Check response with better error handling
            if not response.candidates or response.candidates[0].finish_reason != types.FinishReason.STOP:
                reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
                with print_lock:
                    print(f"      ⚠️  Title slide generation stopped: {reason}")
                if reason == "SAFETY":
                    with print_lock:
                        print(f"      ⚠️  Safety filter triggered. Trying with simplified prompt...")
                    # Could retry with simplified prompt here
                if not response.candidates or not response.candidates[0].content.parts:
                    return False, f"Title slide generation failed: {reason}"
            
            # Extract image from response
            image_saved = False
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Save original bytes first
                    title_image_png = temp_dir / "title_slide.png"
                    with open(title_image_png, 'wb') as f:
                        f.write(part.inline_data.data)
                    # Compress to JPEG to reduce file size
                    title_image_jpg = temp_dir / "title_slide.jpg"
                    compressed_path = _compress_image_to_jpeg(title_image_png, title_image_jpg)
                    # Prefer compressed path
                    generated_images.append(compressed_path)
                    # Remove original PNG to save space if different
                    try:
                        if title_image_png.exists() and compressed_path != title_image_png:
                            title_image_png.unlink()
                    except Exception:
                        pass
                    image_saved = True
                    with print_lock:
                        print(f"      ✓ Generated title slide image")
                    break
            
            if not image_saved:
                return False, "Title slide generation failed: No image data in response"
                
        except Exception as e:
            error_msg = str(e)
            with print_lock:
                print(f"      ❌ Title slide generation failed: {error_msg[:150]}")
            
            # Better error handling
            if "429" in str(e) or "Quota" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                with print_lock:
                    print(f"      ⏳ Rate limit/quota hit, waiting 60 seconds...")
                time.sleep(60)
            elif "403" in str(e) or "PERMISSION_DENIED" in str(e):
                return False, "Permission denied. Check your API key and access."
            elif "SAFETY" in str(e):
                return False, "Safety filter triggered. Try with different content."
            
            return False, f"Title slide generation failed: {error_msg[:200]}"
        
        # Generate content slide images
        for idx, slide in enumerate(slides, 1):
            slide_title = slide.get("title", topic_title)
            slide_content = slide.get("content", [])
            image_prompt = slide.get("image_prompt", "")
            
            # Format content text
            def _clean_text(raw: str) -> str:
                text = raw.strip()
                text = re.sub(r"^\s*(#+|\*+|-+|\d+[\.\)])\s*", "", text)
                text = re.sub(r"[*_`]+", "", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text

            def _strip_metadata(raw: str) -> str:
                text = _clean_text(raw)
                if subject:
                    text = re.sub(re.escape(subject), "", text, flags=re.IGNORECASE)
                text = re.sub(r"\b(клас|class|grade)\s*\d+\b", "", text, flags=re.IGNORECASE)
                text = re.sub(r"\b\d+\s*(клас|class|grade)\b", "", text, flags=re.IGNORECASE)
                text = re.sub(r"\s+", " ", text).strip(" -–—,:;")
                return text

            slide_title = _strip_metadata(slide_title) or topic_title
            cleaned_items = []
            for item in slide_content:
                if not item.strip():
                    continue
                cleaned = _strip_metadata(item)
                if cleaned:
                    cleaned_items.append(cleaned)

            content = " ".join([f"• {item}" for item in cleaned_items])
            
            # Create prompt for complete slide with text embedded
            # STRICTLY enforce subject theme and language
            if image_prompt:
                # Combine image_prompt with subject theme to ensure consistency
                visuals = f"{image_prompt}. Additionally, incorporate {subject_theme} to maintain strict subject consistency."
                slide_prompt = (
                    f"Create a high-quality educational presentation slide. "
                    f"VISUALS: {visuals} STRICTLY follow {subject} theme - do not mix themes from other subjects. "
                    f"LAYOUT: Modern, clean, readable font, high contrast. "
                    f"TEXT: Render ONLY this content in {text_language}, with no labels or UI words like 'headline', 'header', 'body', 'title': "
                    f"Title: '{slide_title}'. Body: '{content}'. "
                    f"Do NOT include words Slide/Слайд or any numbering. "
                    f"Do NOT use markdown or list markers (#, ##, *, -, 1.). "
                    f"Do NOT render JSON keys or labels like name/subject. "
                    f"Do NOT show subject name or grade/class number anywhere. "
                    f"Theme MUST be strictly {subject} themed - no mixing with other subjects."
                )
            else:
                # Use subject theme STRICTLY
                slide_prompt = (
                    f"Create a high-quality educational presentation slide. "
                    f"VISUALS: {subject_theme} STRICTLY follow this theme - do not mix themes from other subjects. "
                    f"LAYOUT: Modern, clean, readable font, high contrast. "
                    f"TEXT: Render ONLY this content in {text_language}, with no labels or UI words like 'headline', 'header', 'body', 'title': "
                    f"Title: '{slide_title}'. Body: '{content}'. "
                    f"Do NOT include words Slide/Слайд or any numbering. "
                    f"Do NOT use markdown or list markers (#, ##, *, -, 1.). "
                    f"Do NOT render JSON keys or labels like name/subject. "
                    f"Do NOT show subject name or grade/class number anywhere. "
                    f"CRITICAL: All text MUST be in {text_language} language. Ensure spelling is correct and text is legible on the background. "
                    f"Theme MUST be strictly {subject} themed - no mixing with other subjects."
                )
            
            try:
                with print_lock:
                    print(f"      🎨 Generating slide {idx}...")
                
                response = client.models.generate_content(
                    model=model_id,
                    contents=slide_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        image_config=types.ImageConfig(
                            aspect_ratio="16:9",
                            image_size="2K",
                        ),
                    ),
                )
                
                # Check response with better error handling
                if not response.candidates or response.candidates[0].finish_reason != types.FinishReason.STOP:
                    reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
                    with print_lock:
                        print(f"      ⚠️  Slide {idx} generation stopped: {reason}")
                    if reason == "SAFETY":
                        with print_lock:
                            print(f"      ⚠️  Safety filter triggered for slide {idx}")
                    if not response.candidates or not response.candidates[0].content.parts:
                        continue
                
                # Extract image from response
                image_saved = False
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        slide_image_png = temp_dir / f"slide_{idx}.png"
                        with open(slide_image_png, 'wb') as f:
                            f.write(part.inline_data.data)
                        # Compress to JPEG to reduce file size
                        slide_image_jpg = temp_dir / f"slide_{idx}.jpg"
                        compressed_path = _compress_image_to_jpeg(slide_image_png, slide_image_jpg)
                        generated_images.append(compressed_path)
                        # Remove original PNG to save space if different
                        try:
                            if slide_image_png.exists() and compressed_path != slide_image_png:
                                slide_image_png.unlink()
                        except Exception:
                            pass
                        image_saved = True
                        with print_lock:
                            print(f"      ✓ Generated slide {idx} image")
                        break
                
                if not image_saved:
                    with print_lock:
                        print(f"      ⚠️  Slide {idx}: No image data in response")
                    continue
                
                # Add delay between slides to avoid rate limits
                if image_saved:
                    time.sleep(10)  # 10 second pause for quota safety
                    
            except Exception as e:
                error_msg = str(e)
                with print_lock:
                    if "429" in str(e) or "Quota" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(f"      ⚠️  Slide {idx} generation failed (rate limit/quota): {error_msg[:80]}")
                        print(f"      ⏳ Waiting 60 seconds...")
                        time.sleep(60)
                    elif "403" in str(e) or "PERMISSION_DENIED" in str(e):
                        print(f"      ⚠️  Slide {idx} generation failed (permissions): {error_msg[:80]}")
                    elif "SAFETY" in str(e):
                        print(f"      ⚠️  Slide {idx} generation failed (safety filter): {error_msg[:80]}")
                    else:
                        print(f"      ⚠️  Slide {idx} generation failed: {error_msg[:100]}")
                # Continue with other slides even if one fails
                time.sleep(2)  # Small delay between slides
        
        # Check if any images were generated
        if len(generated_images) == 0:
            # Clean up
            try:
                temp_dir.rmdir()
            except Exception:
                pass
            return False, "No slide images were generated. Presentation not saved."
        
        # Step 3: Create PDF from generated slide images (Landscape orientation)
        buffer = io.BytesIO()
        
        # Use landscape A4 (width > height)
        landscape_a4 = landscape(A4)  # (11.69 inch, 8.27 inch)
        
        # Use small margins to ensure images fit within frame
        # ReportLab has internal frame calculations, so we need some margin
        margin_pts = 10  # Small margin in points
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape_a4,
            rightMargin=margin_pts,
            leftMargin=margin_pts,
            topMargin=margin_pts,
            bottomMargin=margin_pts
        )
        
        story = []
        # Leave extra headroom so ReportLab never overflows the frame
        max_width_pts = doc.width * 0.95
        max_height_pts = doc.height * 0.95
        safety_scale = 0.95
        
        # Add all generated images as full-page slides in landscape
        for img_path in generated_images:
            if img_path.exists():
                try:
                    # Get actual image dimensions
                    pil_img = PILImage.open(str(img_path))
                    img_width_px, img_height_px = pil_img.size
                    img_aspect = img_width_px / img_height_px
                    
                    # Fit within the available frame while maintaining aspect ratio
                    target_width_pts = max_width_pts
                    target_height_pts = target_width_pts / img_aspect
                    if target_height_pts > max_height_pts:
                        target_height_pts = max_height_pts
                        target_width_pts = max_height_pts * img_aspect
                    
                    target_width_pts *= safety_scale
                    target_height_pts *= safety_scale
                    
                    img = Image(str(img_path))
                    img.drawWidth = target_width_pts
                    img.drawHeight = target_height_pts
                    # Extra clamp to frame in case of rounding differences
                    img._restrictSize(max_width_pts * safety_scale, max_height_pts * safety_scale)
                    story.append(img)
                    story.append(PageBreak())
                except Exception as e:
                    with print_lock:
                        print(f"      ⚠️  Error adding image {img_path.name}: {str(e)[:100]}")
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary images
        for img_path in generated_images:
            try:
                if img_path.exists():
                    img_path.unlink()
            except Exception:
                pass
        try:
            temp_dir.rmdir()
        except Exception:
            pass
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        if len(pdf_bytes) == 0:
            return False, "Generated PDF is empty"
        
        return True, pdf_bytes
        
    except ImportError as e:
        missing_lib = str(e)
        if "reportlab" in missing_lib:
            return False, f"Required library not installed: reportlab. Install with: pip install reportlab"
        elif "PIL" in missing_lib or "Pillow" in missing_lib:
            return False, f"Required library not installed: Pillow. Install with: pip install Pillow"
        return False, f"Required library not installed: {e}"
    except Exception as e:
        return False, f"Exception generating PDF: {e}"


def build_presentation_prompt(
    topic_title: str,
    subject: str,
    form: int,
    context: Optional[str] = None,
) -> str:
    """Build the prompt for Gemini to generate presentation text."""
    
    # Determine language based on subject
    is_english_subject = subject.lower() in ["english", "англійська", "англійська_мова"]
    presentation_language = "English" if is_english_subject else "Ukrainian"
    language_instruction = (
        "English" if is_english_subject 
        else "Ukrainian (українською мовою)"
    )
    
    prompt = f"""Ти - експерт зі створення навчальних презентацій для українських шкільних підручників.

ТЕМА/ПІДТЕМА: {topic_title}
ПРЕДМЕТ: {subject}
КЛАС: {form}

КРИТИЧНО ВАЖЛИВО - МОВА:
- ВСІЙ текст презентації МАЄ бути СТРОГО {presentation_language} мовою
- Якщо предмет НЕ англійська мова - ВСЕ має бути українською мовою
- Якщо предмет - англійська мова - ВСЕ має бути англійською мовою
- ЗАБОРОНЕНО змішувати мови або використовувати англійську для неанглійських предметів
- Текст має бути {language_instruction}

КРИТИЧНО ВАЖЛИВО - ПЕРЕВІРКА НАВЧАЛЬНОГО МАТЕРІАЛУ:
Спочатку обов'язково перевір, чи ця тема/підтема стосується НАВЧАЛЬНОГО МАТЕРІАЛУ (навчального контенту).

ТЕМА ПІДХОДИТЬ для створення презентації, якщо вона:
- Містить навчальний матеріал, який учні вивчають (правила, поняття, теорію, факти, формули, історичні події тощо)
- Стосується конкретної навчальної теми з підручника
- Може бути представлена у вигляді навчальної презентації

ТЕМА НЕ ПІДХОДИТЬ для створення презентації, якщо вона:
- НЕ стосується навчального матеріалу (наприклад: "ЗМІСТ", "Вступ", "Від автора", "Передмова", "Додатки", "Список літератури")
- Це привітання, звернення або адреса до учнів (наприклад: "Шановні дев'ятикласники та дев'ятикласниці!", "Дорогі учні!", "Шановні читачі!", "Дорогі друзі!", будь-які звернення до студентів/учнів)
- Це особисті повідомлення від авторів до читачів
- Це технічні елементи (номери сторінок, заголовки розділів без змісту)
- Це мета-інформація про підручник (автори, рік видання, редакція)
- Це інструкції або методичні рекомендації для вчителів
- Це практичні завдання без теоретичного змісту (наприклад: "Виконайте вправу", "Намалюйте", "Складіть план")
- Це загальні фрази без конкретного навчального змісту

!!! ЗАБОРОНЕНО створювати презентації для:
- Привітань та звернень до учнів (наприклад: "Шановні дев'ятикласники та дев'ятикласниці!", "Дорогі учні!", "Шановні читачі!")
- Особистих повідомлень від авторів
- Будь-яких звернень до читачів/студентів/учнів
- Вступів, передмов, змістів, додатків

ЗАВДАННЯ:
1. ПЕРШИМ КРОКОМ перевір, чи тема стосується НАВЧАЛЬНОГО МАТЕРІАЛУ.
2. Якщо тема є привітанням, зверненням, вступом, або НЕ стосується навчального матеріалу, ОБОВ'ЯЗКОВО поверни порожній текст або повідомлення про те, що тема не підходить.
3. Якщо тема стосується навчального матеріалу, створи текст для навчальної презентації.

!!! КРИТИЧНО ВАЖЛИВО - ТІЛЬКИ ФАКТИ, БЕЗ ВИГАДОК:
- ВСІЙ текст презентації МАЄ бути заснований ТІЛЬКИ на ФАКТАХ з навчального матеріалу
- ЗАБОРОНЕНО вигадувати, придумувати або створювати власний контент
- ЗАБОРОНЕНО використовувати вигадані приклади, фіктивні дані, неіснуючі факти
- ЗАБОРОНЕНО створювати власні сценарії, приклади або ситуації
- ВСІ інформація має бути заснована на РЕАЛЬНИХ фактах, які зазвичай викладаються в підручниках з цієї теми
- Якщо ти не впевнений у фактах для теми, краще поверни порожній текст замість вигадування

ВИМОГИ ДО ПРЕЗЕНТАЦІЇ:
- Презентація має бути структурованою та логічною
- Текст має бути адаптований для навчальної презентації (короткі, зрозумілі речення)
- Презентація має охоплювати основні аспекти теми "{topic_title}"
- Текст має бути СТРОГО {presentation_language} мовою та відповідати рівню {form} класу
- КРИТИЧНО: ВСІЙ текст МАЄ бути СТРОГО {presentation_language} мовою - ЗАБОРОНЕНО використовувати іншу мову
- ВСІЙ текст МАЄ бути заснований ТІЛЬКИ на ФАКТАХ, без вигадок
- Презентація має містити вступ, основну частину з ключовими пунктами, та висновки

ФОРМАТ ВІДПОВІДІ:
Поверни структурований текст для презентації, який можна використати для створення слайдів.
Текст має бути чітким, структурованим та готовим для перетворення в презентацію.

ВАЖЛИВО:
- ОБОВ'ЯЗКОВО перевір, чи тема стосується НАВЧАЛЬНОГО МАТЕРІАЛУ перед створенням презентації
- Якщо тема НЕ стосується навчального матеріалу, поверни порожній текст або повідомлення
- НІКОЛИ не створюй презентації для привітань, звернень до учнів
- НІКОЛИ не створюй презентації для вступів, передмов, звернень від авторів, змістів
- НЕ створюй презентації для технічних елементів, мета-інформації, інструкцій або загальних фраз
- КРИТИЧНО: ВСІЙ текст МАЄ бути заснований ТІЛЬКИ на ФАКТАХ з навчального матеріалу
- ЗАБОРОНЕНО вигадувати, придумувати або створювати власний контент, приклади, дані або факти
- Якщо не впевнений у фактах для теми, краще поверни порожній текст замість вигадування
- Текст має бути якісним та корисним для навчання"""
    
    if context:
        prompt += f"\n\nКОНТЕКСТ (для кращого розуміння теми):\n{context}"
    
    return prompt


def extract_subject_form_from_path(json_path: Path) -> Tuple[Optional[str], Optional[int]]:
    """Extract subject and form from the JSON file path.
    
    Expected path: toc_openai_filtered/{subject}/{form}/{year}/{book_name}/{filename}.json
    """
    parts = json_path.parts
    try:
        # Find toc_openai_filtered or toc_openai in the path (support both for compatibility)
        toc_idx = None
        for i, part in enumerate(parts):
            if part == "toc_openai_filtered" or part == "toc_openai":
                toc_idx = i
                break
        
        if toc_idx is None or toc_idx + 2 >= len(parts):
            return None, None
        
        subject = parts[toc_idx + 1]
        form_str = parts[toc_idx + 2]
        
        # Try to extract form number
        form_match = re.search(r'\d+', form_str)
        if form_match:
            form = int(form_match.group())
        else:
            form = None
        
        return subject, form
    except (IndexError, ValueError):
        return None, None


def generate_presentation_text_for_topic(
    topic_title: str,
    subject: str,
    form: int,
    context: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Generate presentation text for a topic using Gemini on Vertex AI."""
    prompt = build_presentation_prompt(topic_title, subject, form, context)
    
    for attempt in range(MAX_API_RETRIES):
        try:
            client, types = _init_vertex_genai_client()
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT"],
                ),
            )
            response_text = getattr(response, "text", None) or str(response)
            if not response_text or not response_text.strip():
                if attempt < MAX_API_RETRIES - 1:
                    time.sleep(API_RETRY_BASE_DELAY * (attempt + 1))
                    continue
                return False, "Empty response from Gemini"
            
            # Check if Gemini indicated the topic is not suitable
            response_lower = response_text.lower()
            if any(phrase in response_lower for phrase in ["не підходить", "не стосується", "не можу", "порожній"]):
                return False, "Topic not suitable for presentation"
            
            return True, response_text.strip()
                
        except Exception as e:
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(API_RETRY_BASE_DELAY * (attempt + 1))
                continue
            return False, f"Exception: {e}"
    
    return False, "Max retries exceeded"


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Sanitize text to be used as a filename."""
    # Remove or replace invalid characters
    text = re.sub(r'[<>:"/\\|?*]', '_', text)
    text = re.sub(r'\s+', '_', text)
    text = text.strip('._')
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text or "untitled"


def process_single_topic(
    topic_title: str,
    subject: str,
    form: int,
    output_dir: Path,
) -> Tuple[str, bool, Optional[str]]:
    """Process a single topic and generate presentation.
    
    Returns (topic_title, success, message)
    """
    # Skip generic TOC entries, greetings, and addresses to students
    topic_lower = topic_title.lower()
    skip_patterns = [
        "зміст", "вступ", "від автора", "передмова", "додатки", "список літератури",
        "шановні", "дорогі", "читачі", "друзі",  # Greetings and addresses
        "шановні дев'ятикласники", "шановні дев'ятикласниці",  # Specific greeting pattern
        "дорогі учні", "дорогі студенти", "шановні учні", "шановні студенти",
    ]
    if any(skip_word in topic_lower for skip_word in skip_patterns):
        with print_lock:
            print(f"  ⏭️  Skipping (greeting/intro): {topic_title[:60]}...")
        return (topic_title, False, "greeting/intro")
    
    # Additional check: if topic starts with greeting words, skip it
    greeting_starters = ["шановні", "дорогі", "читачі", "друзі"]
    if any(topic_lower.startswith(starter) for starter in greeting_starters):
        with print_lock:
            print(f"  ⏭️  Skipping (greeting): {topic_title[:60]}...")
        return (topic_title, False, "greeting")
    
    # Check if presentation already exists for this topic
    safe_topic = sanitize_filename(topic_title)
    potential_filenames = [f"{safe_topic}.pdf"]
    # Also check for numbered variants (in case of previous conflicts)
    for i in range(1, 10):  # Check up to 9 variants
        potential_filenames.append(f"{safe_topic}_{i}.pdf")
    
    # Check if any of these files exist
    presentation_exists = False
    existing_file = None
    for filename in potential_filenames:
        potential_path = output_dir / filename
        if potential_path.exists() and potential_path.stat().st_size > 0:
            presentation_exists = True
            existing_file = filename
            break
    
    if presentation_exists:
        with print_lock:
            print(f"  ✓ Presentation already exists: {existing_file} - skipping")
        return (topic_title, False, "already_exists")
    
    with print_lock:
        print(f"  📝 Generating presentation for: {topic_title[:60]}...")
    
    # Step 1: Generate presentation text with Gemini
    ok, presentation_text = generate_presentation_text_for_topic(topic_title, subject, form)
    
    if not ok or not presentation_text:
        with print_lock:
            print(f"    ❌ Error generating text: {presentation_text}")
        return (topic_title, False, f"text_generation_error: {presentation_text}")
    
    # Step 2: Generate PDF presentation with Gemini 3 Image Preview
    ok_pdf, pdf_result = generate_presentation_with_gemini(
        presentation_text, topic_title, subject, form
    )
    
    if not ok_pdf:
        with print_lock:
            print(f"    ❌ Error generating PDF: {pdf_result}")
        return (topic_title, False, f"pdf_generation_error: {pdf_result}")
    
    if not pdf_result or len(pdf_result) == 0:
        with print_lock:
            print(f"    ❌ Empty PDF generated")
        return (topic_title, False, "empty_pdf")
    
    # Generate output filename
    output_filename = f"{safe_topic}.pdf"
    output_path = output_dir / output_filename
    
    # Handle filename conflicts (thread-safe)
    counter = 1
    while output_path.exists():
        output_filename = f"{safe_topic}_{counter}.pdf"
        output_path = output_dir / output_filename
        counter += 1
    
    # Save PDF (thread-safe file writing)
    try:
        with open(output_path, 'wb') as f:
            f.write(pdf_result)
        with print_lock:
            print(f"    ✅ Saved: {output_filename} ({len(pdf_result)} bytes)")
        return (topic_title, True, "success")
    except Exception as e:
        with print_lock:
            print(f"    ❌ Error saving file: {e}")
        return (topic_title, False, f"save_error: {e}")


def process_toc_file(json_path: Path, output_root: Path) -> Tuple[int, int]:
    """Process a single TOC JSON file and generate presentations for each topic/subtopic.
    
    Returns (success_count, skip_count).
    """
    with print_lock:
        print(f"\nProcessing: {json_path}")
    
    # Extract subject and form from path
    subject, form = extract_subject_form_from_path(json_path)
    if not subject or form is None:
        with print_lock:
            print(f"  ⚠️  Could not extract subject/form from path, skipping")
        return 0, 0
    
    with print_lock:
        print(f"  Subject: {subject}, Form: {form}")
    
    # Read TOC JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            toc_data = json.load(f)
    except Exception as e:
        with print_lock:
            print(f"  ❌ Error reading file: {e}")
        return 0, 0
    
    if not isinstance(toc_data, dict) or "toc" not in toc_data:
        with print_lock:
            print(f"  ⚠️  Invalid TOC structure, skipping")
        return 0, 0
    
    toc_items = toc_data.get("toc", [])
    if not toc_items:
        with print_lock:
            print(f"  ⚠️  Empty TOC, skipping")
        return 0, 0
    
    # Create output directory
    output_dir = output_root / subject / str(form)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect topics to process
    topics_to_process = []
    for toc_item in toc_items:
        if not isinstance(toc_item, dict):
            continue
        
        topic_title = toc_item.get("title", "").strip()
        if topic_title:
            topics_to_process.append(topic_title)
    
    if not topics_to_process:
        return 0, 0
    
    with print_lock:
        print(f"  Found {len(topics_to_process)} topics to process (using {NUM_WORKERS} workers)")
    
    # Process topics in parallel
    success_count = 0
    skip_count = 0
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_topic = {
            executor.submit(
                process_single_topic,
                topic_title,
                subject,
                form,
                output_dir,
            ): topic_title
            for topic_title in topics_to_process
        }
        
        # Process completed tasks
        for future in as_completed(future_to_topic):
            topic_title = future_to_topic[future]
            try:
                _, success, message = future.result()
                if success:
                    success_count += 1
                else:
                    skip_count += 1
            except Exception as e:
                with print_lock:
                    print(f"  ❌ Exception processing {topic_title[:60]}: {e}")
                skip_count += 1
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
    
    return success_count, skip_count


def main():
    """Main function to process all TOC JSON files."""
    # Ensure filenames under toc_openai_filtered are safe/short enough
    _shorten_toc_filenames_if_needed()

    if not TOC_ROOT.exists():
        print(f"❌ TOC directory not found: {TOC_ROOT}")
        sys.exit(1)
    
    # Find all JSON files
    json_files = list(TOC_ROOT.rglob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {TOC_ROOT}")
        sys.exit(1)
    
    print(f"Found {len(json_files)} JSON files to process")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Using Gemini model: {GEMINI_MODEL}")
    print(f"Using Vertex image model: {VERTEX_IMAGE_MODEL}")
    print(f"Number of parallel workers: {NUM_WORKERS}")
    
    # Validate Vertex config early
    project, region = _load_vertex_env_defaults()
    print(f"Vertex project: {project}")
    print(f"Vertex region: {region}")
    
    total_success = 0
    total_skip = 0
    
    for i, json_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}]")
        success, skip = process_toc_file(json_path, OUTPUT_ROOT)
        total_success += success
        total_skip += skip
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  ✅ Successfully generated: {total_success} presentations")
    print(f"  ⏭️  Skipped: {total_skip} topics")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
