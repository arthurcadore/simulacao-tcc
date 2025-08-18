import fitz
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from scour import scour
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Diretórios
script_dir = Path(__file__).parent.absolute()
input_dir = script_dir.parent.parent.parent / "assets"
output_dir = script_dir / "assets"
output_dir.mkdir(parents=True, exist_ok=True)

MAX_SVG_SIZE = 4 * 1024 * 1024  # 4 MB
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
        
        # Abre o PDF e obtém a primeira página
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # Obtém as dimensões da página
        width = page.rect.width
        height = page.rect.height
        
        # Cria uma matriz de transformação para garantir toda a página
        # Aumenta um pouco o zoom para garantir que nada seja cortado
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        
        # Primeira tentativa com precisão padrão (8)
        svg_code = page.get_svg_image(matrix=mat, text_as_path=False)
        optimized_svg = optimize_svg(svg_code, digits=8)
        svg_size = len(optimized_svg.encode("utf-8"))
        
        if svg_size > MAX_SVG_SIZE:
            # Reduz zoom até caber em 4 MB
            zoom = 1  # metade do zoom original
            mat = fitz.Matrix(zoom, zoom)
            svg_code = page.get_svg_image(matrix=mat, text_as_path=False)
            optimized_svg = optimize_svg(svg_code, digits=8)
            svg_size = len(optimized_svg.encode("utf-8"))
            msg = f"Convertido com zoom reduzido: {filename} ({svg_size/1024/1024:.2f} MB, zoom={zoom})"

        else:
            msg = f"Convertido normalmente: {filename} ({svg_size/1024/1024:.2f} MB, digits=8)"

        # Salva SVG
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(optimized_svg)
        remove_white_background(svg_path)

        return msg

    except Exception as e:
        return f"Erro ao processar {filename}: {e}"


if __name__ == "__main__":
    pdf_files = [f.name for f in input_dir.glob("*.pdf")]

    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = {executor.submit(process_file, f): f for f in pdf_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Convertendo"):
            print(future.result())
