import os
from pathlib import Path
from PIL import Image, ImageSequence, ImageEnhance, ImageOps
from tqdm import tqdm

script_dir = Path(__file__).parent.absolute()
input_dir = script_dir.parent / "media"
output_dir = script_dir / "api" / "media"
output_dir.mkdir(parents=True, exist_ok=True)


def recolor_frame(frame: Image.Image, mode: str) -> Image.Image:
    """
    Ajusta as cores do frame para modo claro ou escuro.
    - mode: 'light' ou 'dark'
    """
    frame = frame.convert("RGBA")
    r, g, b, a = frame.split()
    rgb = Image.merge("RGB", (r, g, b))

    if mode == "dark":
        # Inverte e realça o contraste — texto branco sobre fundo escuro
        rgb = ImageOps.invert(rgb)
        enhancer = ImageEnhance.Contrast(rgb)
        rgb = enhancer.enhance(1.4)
    elif mode == "light":
        # Leve aumento de brilho e contraste
        enhancer = ImageEnhance.Brightness(rgb)
        rgb = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Contrast(rgb)
        rgb = enhancer.enhance(1.1)

    recolored = Image.merge("RGBA", (*rgb.split(), a))
    return recolored

def process_gif(filename):
    gif_path = input_dir / filename
    base_name = gif_path.stem
    gif_light_path = output_dir / f"{base_name}_light.gif"
    gif_dark_path = output_dir / f"{base_name}_dark.gif"

    try:
        im = Image.open(gif_path)
        frames_light, frames_dark = [], []

        for frame in ImageSequence.Iterator(im):
            frames_light.append(recolor_frame(frame, "light"))
            frames_dark.append(recolor_frame(frame, "dark"))

        duration = im.info.get("duration", 100)
        loop = im.info.get("loop", 0)

        # Salva versões claro/escuro
        frames_light[0].save(
            gif_light_path,
            save_all=True,
            append_images=frames_light[1:],
            duration=duration,
            loop=loop,
            optimize=True,
            disposal=2,
        )

        frames_dark[0].save(
            gif_dark_path,
            save_all=True,
            append_images=frames_dark[1:],
            duration=duration,
            loop=loop,
            optimize=True,
            disposal=2,
        )

        return f"{filename}: gerado {gif_light_path.name} e {gif_dark_path.name}"

    except Exception as e:
        return f"Erro ao processar {filename}: {e}"

if __name__ == "__main__":
    gif_files = [f.name for f in input_dir.glob("*.gif")]
    for result in tqdm(map(process_gif, gif_files), total=len(gif_files), desc="Gerando GIFs adaptados"):
        print(result)