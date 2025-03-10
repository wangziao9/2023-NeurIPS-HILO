from PIL import Image, ImageDraw, ImageFont
import numpy as np

def char_to_image(char, img_size=64, font_path="../assets/Noto_Sans_SC/static/NotoSansSC-Regular.ttf"):
    """
    Convert a Chinese character to a grayscale image represented as a 2D NumPy array.
    Example font can be downloaded from https://fonts.google.com/noto/specimen/Noto+Sans+SC
    
    Parameters:
    - char: str, the Chinese character to render.
    - font_path: str, path to a .ttf or .otf font file supporting Chinese characters.
    - img_size: int, size of the output square image (default 64x64).
    
    Returns:
    - np.array: 2D array (floating point) representing the grayscale image.
    """
    # Load the font
    try:
        font = ImageFont.truetype(font_path, img_size)
    except IOError:
        raise ValueError("Font file not found or unsupported format.")

    # Create a blank white image
    img = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)

    left, top, right, bottom = draw.textbbox((0,0), char, font=font)
    print(f"left={left},top={top}, right={right},bottom={bottom}")
    w = right - left
    h = bottom - top
    x = (img_size - w) // 2
    y = (img_size - h) // 2
    print(f"x={x},y={y}")

    # Draw character in black
    draw.text((x-left,y-top), char, font=font, fill=0)

    # Convert to NumPy array (normalize to 0-1 float)
    np_img = np.array(img).astype(np.float32) / 255.0
    return np_img

# Example usage:
char_image = char_to_image("国", img_size=64)
print(char_image.shape)
import matplotlib.pyplot as plt
plt.imshow(char_image)
plt.show()