import fitz
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from scour import scour
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Obtém o diretório do script atual
script_dir = Path(__file__).parent.absolute()
# Define os diretórios de entrada e saída
input_dir = script_dir.parent / "out"
output_dir = script_dir / "api" / "assets"

# Cria o diretório de saída se não existir
output_dir.mkdir(parents=True, exist_ok=True)

MAX_SVG_SIZE = 4 * 1024 * 1024  # 1 MB
N_CORES = 12


def remove_white_background(svg_path):
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        for elem in list(root):
            if elem.tag.endswith("rect") and elem.attrib.get("fill", "").lower() in ("#ffffff", "white"):
                root.remove(elem)
        tree.write(svg_path)
    except Exception as e:
        print(f"Erro ao limpar fundo do {svg_path}: {e}")


def optimize_svg(svg_code):
    options = scour.sanitizeOptions()
    options.remove_metadata = True
    options.strip_comments = True
    options.simple_colors = True
    options.shorten_ids = True
    options.digits = 8
    return scour.scourString(svg_code, options=options)


def process_file(filename):
    try:
        pdf_path = input_dir / filename
        base_name = os.path.splitext(filename)[0]
        svg_path = output_dir / f"{base_name}.svg"

        doc = fitz.open(pdf_path)
        page = doc[0]

        # Exporta SVG com texto como <text>
        svg_code = page.get_svg_image(text_as_path=False)

        # Otimiza SVG (em memória)
        optimized_svg = optimize_svg(svg_code)
        svg_size = len(optimized_svg.encode("utf-8"))

        if svg_size > MAX_SVG_SIZE:
            # Se já ultrapassar 1MB em memória, gera direto PNG
            zoom = 8
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_path = output_dir / f"{base_name}.png"
            pix.save(png_path)

            return f"Convertido para PNG: {filename} -> {png_path} ({svg_size/1024/1024:.2f} MB em SVG)"
        else:
            # Salva SVG otimizado
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(optimized_svg)

            remove_white_background(svg_path)
            return f"Convertido para SVG: {filename} ({svg_size/1024:.2f} KB)"

    except Exception as e:
        return f"Erro ao processar {filename}: {e}"


if __name__ == "__main__":
    pdf_files = [f.name for f in input_dir.glob("*.pdf")]

    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = {executor.submit(process_file, f): f for f in pdf_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Convertendo"):
            print(future.result())
