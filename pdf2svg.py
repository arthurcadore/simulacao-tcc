import fitz  # PyMuPDF
import os
import xml.etree.ElementTree as ET

input_dir = "./out"
output_dir = "./docs/api/assets"

os.makedirs(output_dir, exist_ok=True)

def remove_white_background(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    for elem in list(root):
        # Remove retângulo de fundo branco
        if elem.tag.endswith("rect") and elem.attrib.get("fill", "").lower() in ("#ffffff", "white"):
            root.remove(elem)
    tree.write(svg_path)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_dir, filename)
        svg_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".svg")

        doc = fitz.open(pdf_path)
        page = doc[0]  # primeira página
        svg_code = page.get_svg_image()  # gera SVG vetorial
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_code)
        remove_white_background(svg_path)
        print(f"Convertido: {filename} -> {svg_path}")
