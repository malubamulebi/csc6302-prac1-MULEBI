import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def unsharp_mask(image, amount, kernel_size=5):
    """
    Sharpen an image using:
    sharpened = original + amount * (original - blurred)
    """
    original_float = image.astype(np.float32)

    blurred = cv2.GaussianBlur(
        original_float,
        (kernel_size, kernel_size),
        sigmaX=0
    )

    detail_mask = original_float - blurred

    sharpened = original_float + amount * detail_mask

    # Keep pixel values inside the valid 8-bit image range.
    sharpened = np.clip(sharpened, 0, 255)

    return sharpened.astype(np.uint8), blurred.astype(np.uint8), detail_mask


def main():
    parser = argparse.ArgumentParser(
        description="Task 3: sharpening with unsharp masking"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/task3")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(input_path))

    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    amounts = [0.5, 1.0, 2.0]
    sharpened_results = {}

    for amount in amounts:
        sharpened_bgr, blurred_bgr, detail_mask = unsharp_mask(
            image_bgr,
            amount=amount,
            kernel_size=5
        )

        sharpened_results[amount] = cv2.cvtColor(
            sharpened_bgr,
            cv2.COLOR_BGR2RGB
        )

        amount_name = str(amount).replace(".", "_")

        cv2.imwrite(
            str(
                output_directory
                / f"sharpened_amount_{amount_name}.png"
            ),
            sharpened_bgr
        )

    cv2.imwrite(
        str(output_directory / "blurred_image.png"),
        blurred_bgr
    )

    # Convert the detail mask into a visible image for display.
    visible_detail = cv2.normalize(
        detail_mask,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(
        str(output_directory / "detail_mask.png"),
        visible_detail
    )

    figure, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")

    for index, amount in enumerate(amounts, start=1):
        axes[index].imshow(sharpened_results[amount])
        axes[index].set_title(f"Amount = {amount}")

    for axis in axes:
        axis.axis("off")

    figure.suptitle("Unsharp Masking Results")
    figure.tight_layout()

    figure.savefig(
        output_directory / "unsharp_comparison.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)

    print("Task 3 completed successfully.")
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    main()