import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_pink_mask(hsv_image):
    """Create a binary mask for saturated pink and magenta pixels."""
    lower_pink = np.array([155, 55, 55], dtype=np.uint8)
    upper_pink = np.array([179, 255, 255], dtype=np.uint8)

    return cv2.inRange(hsv_image, lower_pink, upper_pink)


def save_channels(original_rgb, converted_image, channel_names, title, output_path):
    """Display and save the original image and three colour-space channels."""
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original RGB")

    for index, channel_name in enumerate(channel_names):
        axes[index + 1].imshow(
            converted_image[:, :, index],
            cmap="gray"
        )
        axes[index + 1].set_title(channel_name)

    for axis in axes:
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Task 1: colour spaces and HSV colour picker"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/task1")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(input_path))

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    save_channels(
        image_rgb,
        image_hsv,
        ["Hue (H)", "Saturation (S)", "Value (V)"],
        "HSV Colour-Space Channels",
        output_directory / "hsv_channels.png"
    )

    save_channels(
        image_rgb,
        image_lab,
        ["Lightness (L)", "Green-Red (a)", "Blue-Yellow (b)"],
        "LAB Colour-Space Channels",
        output_directory / "lab_channels.png"
    )

    pink_mask = create_pink_mask(image_hsv)

    isolated_pink = cv2.bitwise_and(
        image_bgr,
        image_bgr,
        mask=pink_mask
    )

    cv2.imwrite(
        str(output_directory / "pink_mask.png"),
        pink_mask
    )

    cv2.imwrite(
        str(output_directory / "pink_isolated.png"),
        isolated_pink
    )

    figure, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")

    axes[1].imshow(pink_mask, cmap="gray")
    axes[1].set_title("HSV Binary Mask")

    axes[2].imshow(
        cv2.cvtColor(isolated_pink, cv2.COLOR_BGR2RGB)
    )
    axes[2].set_title("Isolated Pink")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(
        output_directory / "colour_picker_result.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(figure)

    print("Task 1 completed successfully.")
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    main()