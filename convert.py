import os
from PIL import Image
import numpy as np

def convert_binary_images_to_indexed(
    input_dir="Datasets_CrackNex/LCSD/SegmentationClassOLD/",
    output_dir="Datasets_CrackNex/LCSD/SegmentationClass/"
):
    """
    Convert all PNG images in 'input_dir' so that 255-valued pixels
    are mapped either to class 1 (if index < 227) or to class 2 (if index >= 227).
    The background remains 0. Then save them as palettized 8-bit images to 'output_dir'.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a palette:
    #  - index 0 -> (  0,   0,   0) = black (background)
    #  - index 1 -> (255, 255, 255) = white (normal-light object)
    #  - index 2 -> (255,   0,   0) = red   (low-light object)
    # Extend the palette to 256 * 3 entries
    palette = [
        0,   0,   0,   # index 0
        255, 255, 255, # index 1
        255,   0,   0  # index 2
    ]
    palette += [0, 0, 0] * (256 - 3)

    # Loop over all PNG images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Parse the numeric index from file name "LCSD_XXXX.png"
            base_name = os.path.splitext(filename)[0]  # e.g., "LCSD_0095"
            try:
                index_str = base_name.split('_')[1]     # e.g., "0095"
                index_val = int(index_str)
            except (IndexError, ValueError):
                # If the file name doesn't follow LCSD_XXXX, skip
                print(f"Skipping {filename}: cannot parse index.")
                continue

            # Decide which class should replace 255 based on index
            if index_val < 227:
                object_class = 1  # normal light
            else:
                object_class = 2  # low light

            # 1. Open image and convert to grayscale (1 channel)
            img = Image.open(input_path).convert("L")

            # 2. Convert to numpy array
            data = np.array(img, dtype=np.uint8)

            # 3. Map 255 -> object_class; 0 stays 0
            data[data == 255] = object_class

            # 4. Convert back to PIL Image, in mode 'P' for palettized
            indexed_img = Image.fromarray(data, mode="P")

            # 5. Apply the custom palette
            indexed_img.putpalette(palette)

            # 6. Save the result
            indexed_img.save(output_path, optimize=True)
            print(f"Converted and saved: {output_path}")

if __name__ == "__main__":
    convert_binary_images_to_indexed()