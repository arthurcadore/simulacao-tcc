import fitz
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from scour import scour
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Diretórios
script_dir = Path(__file__).parent.absolute()
input_dir = script_dir / "out"
output_dir = script_dir / "assets"
output_dir.mkdir(parents=True, exist_ok=True)

MAX_SVG_SIZE = 4 * 1024 * 1024  # 4 MB
N_CORES = 12

def remove_white_background(svg_path):
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Remove fundos brancos (rectangles com fill branco)
        for elem in list(root):
            if elem.tag.endswith("rect"):
                fill = elem.attrib.get("fill", "").lower()
                style = elem.attrib.get("style", "").lower()
                if (
                    "white" in fill
                    or "#ffffff" in fill
                    or "fill:#ffffff" in style
                    or "fill:white" in style
                ):
                    root.remove(elem)

        # Remove fills brancos de textos e tspan e força classe 'fill'
        for elem in root.iter():
            if elem.tag.endswith("text") or elem.tag.endswith("tspan"):
                fill = elem.attrib.get("fill", "").lower()
                style = elem.attrib.get("style", "").lower()
                if fill in ("white", "#ffffff"):
                    del elem.attrib["fill"]
                if "fill:white" in style or "fill:#ffffff" in style:
                    new_style = ";".join(
                        part for part in style.split(";") if "fill:white" not in part and "fill:#ffffff" not in part
                    )
                    if new_style:
                        elem.attrib["style"] = new_style
                    else:
                        elem.attrib.pop("style", None)
                prev_class = elem.attrib.get("class", "")
                elem.attrib["class"] = f"{prev_class} fill".strip()

        tree.write(svg_path)

    except Exception as e:
        print(f"Erro ao patchar {svg_path}: {e}")


def patch_svg(svg_path):
    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Bloco de estilo CSS
        style_element = ET.fromstring("""
        <style>
          .fill { fill: black; }
          .stroke { stroke: black; }
          @media (prefers-color-scheme: dark) {
            .fill { fill: white; }
            .stroke { stroke: white; }
          }
        </style>
        """)
        root.insert(0, style_element)

        def is_black_rgb(value: str) -> bool:
            return value.strip().lower() in {
                "black", "#000", "#000000",
                "rgb(0%, 0%, 0%)", "rgb(0,0,0)", "rgb(0, 0, 0)"
            }

        for elem in root.iter():
            classes = []
            is_legend = False
            if elem.tag.endswith("text"):
                font_family = elem.attrib.get("font-family", "").upper()
                if "CMR12" in font_family:
                    is_legend = True

            fill = elem.attrib.get("fill")
            if fill and is_black_rgb(fill) and not is_legend:
                del elem.attrib["fill"]
                classes.append("fill")

            stroke = elem.attrib.get("stroke")
            if stroke and is_black_rgb(stroke):
                del elem.attrib["stroke"]
                classes.append("stroke")

            style = elem.attrib.get("style", "").lower()
            new_style_parts = []
            for part in style.split(";"):
                part = part.strip()
                if part.startswith("fill:") and ("#000" in part or "black" in part) and not is_legend:
                    classes.append("fill")
                    continue
                if part.startswith("stroke:") and ("#000" in part or "black" in part):
                    classes.append("stroke")
                    continue
                if part:
                    new_style_parts.append(part)

            if new_style_parts:
                elem.attrib["style"] = ";".join(new_style_parts)
            elif "style" in elem.attrib:
                del elem.attrib["style"]

            if classes:
                prev_class = elem.attrib.get("class", "")
                elem.attrib["class"] = f"{prev_class} {' '.join(classes)}".strip()

        tree.write(svg_path)

    except Exception as e:
        print(f"Erro ao aplicar patch no {svg_path}: {e}")


def optimize_svg(svg_code, digits=8):
    options = scour.sanitizeOptions()
    options.remove_metadata = True
    options.strip_comments = True
    options.simple_colors = True
    options.shorten_ids = True
    options.digits = digits
    return scour.scourString(svg_code, options=options)


def process_file(filename):
    try:
        pdf_path = input_dir / filename
        base_name = os.path.splitext(filename)[0]
        svg_path = output_dir / f"{base_name}.svg"
        
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        
        svg_code = page.get_svg_image(matrix=mat, text_as_path=False)
        optimized_svg = optimize_svg(svg_code, digits=8)
        svg_size = len(optimized_svg.encode("utf-8"))
        
        if svg_size > MAX_SVG_SIZE:
            zoom = 1
            mat = fitz.Matrix(zoom, zoom)
            svg_code = page.get_svg_image(matrix=mat, text_as_path=False)
            optimized_svg = optimize_svg(svg_code, digits=8)
            svg_size = len(optimized_svg.encode("utf-8"))
            msg = f"Convertido com zoom reduzido: {filename} ({svg_size/1024/1024:.2f} MB, zoom={zoom})"
        else:
            msg = f"Convertido normalmente: {filename} ({svg_size/1024/1024:.2f} MB, digits=8)"

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(optimized_svg)

        remove_white_background(svg_path)
        patch_svg(svg_path)

        return msg

    except Exception as e:
        return f"Erro ao processar {filename}: {e}"


if __name__ == "__main__":
    pdf_files = [f.name for f in input_dir.glob("*.pdf")]

    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = {executor.submit(process_file, f): f for f in pdf_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Convertendo"):
            print(future.result())
