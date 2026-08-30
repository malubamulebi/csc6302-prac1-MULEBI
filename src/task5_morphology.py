import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def resize_image(image, maximum_dimension=800):
    """Resize a large image while keeping its aspect ratio."""
    if max(image.shape[:2]) <= maximum_dimension:
        return image

    scale = maximum_dimension / max(image.shape[:2])

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )


def create_object_mask(image):
    """
    Threshold the image in LAB space.

    The b channel separates the beige/yellow bottle from the
    grey background, while L separates the bright white bottle.
    """
    lab_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    lightness, _, blue_yellow = cv2.split(lab_image)

    yellow_object = blue_yellow > 140
    bright_object = lightness > 185

    binary_mask = np.where(
        yellow_object | bright_object,
        255,
        0
    ).astype(np.uint8)

    # Ignore thin image-border regions that are not objects.
    vertical_margin = int(binary_mask.shape[0] * 0.04)
    horizontal_margin = int(binary_mask.shape[1] * 0.04)

    binary_mask[:vertical_margin, :] = 0
    binary_mask[-vertical_margin:, :] = 0
    binary_mask[:, :horizontal_margin] = 0
    binary_mask[:, -horizontal_margin:] = 0

    return binary_mask


def main():
    parser = argparse.ArgumentParser(
        description="Task 5: morphology and object counting"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/task5")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    original_image = cv2.imread(str(input_path))

    if original_image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    image = resize_image(original_image)

    initial_mask = create_object_mask(image)

    # A 9x9 ellipse follows the bottles' rounded boundaries.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    eroded_mask = cv2.erode(
        initial_mask,
        kernel,
        iterations=1
    )

    dilated_mask = cv2.dilate(
        eroded_mask,
        kernel,
        iterations=1
    )

    opened_mask = cv2.morphologyEx(
        initial_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    closed_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    cv2.imwrite(
        str(output_directory / "01_initial_mask.png"),
        initial_mask
    )
    cv2.imwrite(
        str(output_directory / "02_eroded_mask.png"),
        eroded_mask
    )
    cv2.imwrite(
        str(output_directory / "03_dilated_mask.png"),
        dilated_mask
    )
    cv2.imwrite(
        str(output_directory / "04_opened_mask.png"),
        opened_mask
    )
    cv2.imwrite(
        str(output_directory / "05_closed_mask.png"),
        closed_mask
    )

    number_of_labels, labels, statistics, centroids = (
        cv2.connectedComponentsWithStats(
            closed_mask,
            connectivity=8
        )
    )

    annotated_image = image.copy()
    object_count = 0

    image_area = image.shape[0] * image.shape[1]
    minimum_area = image_area * 0.015
    minimum_height = image.shape[0] * 0.20

    for label_number in range(1, number_of_labels):
        x = statistics[label_number, cv2.CC_STAT_LEFT]
        y = statistics[label_number, cv2.CC_STAT_TOP]
        width = statistics[label_number, cv2.CC_STAT_WIDTH]
        height = statistics[label_number, cv2.CC_STAT_HEIGHT]
        area = statistics[label_number, cv2.CC_STAT_AREA]

        # Reject small pieces of remaining background texture.
        if area >= minimum_area and height >= minimum_height:
            object_count += 1

            cv2.rectangle(
                annotated_image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                3
            )

            centre_x, centre_y = centroids[label_number]

            cv2.putText(
                annotated_image,
                f"Object {object_count}",
                (int(centre_x) - 45, int(centre_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    cv2.imwrite(
        str(output_directory / "counted_objects.png"),
        annotated_image
    )

    masks = [
        ("Initial threshold", initial_mask),
        ("Erosion", eroded_mask),
        ("Dilation", dilated_mask),
        ("Opening", opened_mask),
        ("Closing", closed_mask)
    ]

    figure, axes = plt.subplots(2, 3, figsize=(13, 9))

    axes[0, 0].imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )
    axes[0, 0].set_title("Original")

    for axis, (title, mask) in zip(
        axes.flat[1:],
        masks
    ):
        axis.imshow(mask, cmap="gray")
        axis.set_title(title)

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(
        "Morphological Mask-Cleaning Stages"
    )
    figure.tight_layout()

    figure.savefig(
        output_directory / "morphology_steps.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)

    result_figure, result_axes = plt.subplots(
        1,
        2,
        figsize=(10, 6)
    )

    result_axes[0].imshow(
        closed_mask,
        cmap="gray"
    )
    result_axes[0].set_title("Final Cleaned Mask")

    result_axes[1].imshow(
        cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB
        )
    )
    result_axes[1].set_title(
        f"Connected Components: {object_count}"
    )

    for axis in result_axes:
        axis.axis("off")

    result_figure.tight_layout()

    result_figure.savefig(
        output_directory / "object_count_result.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(result_figure)

    print("Task 5 completed successfully.")
    print(f"Object count: {object_count}")
    print("Structuring element: 9x9 ellipse")
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    main()