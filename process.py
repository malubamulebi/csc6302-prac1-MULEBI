import argparse
from pathlib import Path

import cv2
import numpy as np


def validate_kernel_size(kernel_size):
    """Ensure the kernel size is a positive odd number."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            "Kernel size must be an odd number of at least 3."
        )


def colour_picker(image):
    """Isolate saturated pink and magenta pixels using HSV."""
    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_pink = np.array(
        [155, 55, 55],
        dtype=np.uint8
    )

    upper_pink = np.array(
        [179, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv_image,
        lower_pink,
        upper_pink
    )

    return cv2.bitwise_and(
        image,
        image,
        mask=mask
    )


def denoise_image(image, kernel_size):
    """Apply median filtering."""
    validate_kernel_size(kernel_size)

    return cv2.medianBlur(
        image,
        kernel_size
    )


def sharpen_image(image, kernel_size, amount):
    """Apply unsharp masking."""
    validate_kernel_size(kernel_size)

    original_float = image.astype(np.float32)

    blurred = cv2.GaussianBlur(
        original_float,
        (kernel_size, kernel_size),
        sigmaX=0
    )

    detail_mask = original_float - blurred

    sharpened = (
        original_float
        + amount * detail_mask
    )

    return np.clip(
        sharpened,
        0,
        255
    ).astype(np.uint8)


def detect_edges(image):
    """Apply Canny edge detection."""
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Canny(
        grayscale,
        100,
        200
    )


def count_objects(image, kernel_size):
    """Create a cleaned mask and count large components."""
    validate_kernel_size(kernel_size)

    maximum_dimension = 800

    if max(image.shape[:2]) > maximum_dimension:
        scale = maximum_dimension / max(image.shape[:2])

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    lab_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    lightness, _, blue_yellow = cv2.split(
        lab_image
    )

    mask = np.where(
        (blue_yellow > 140) | (lightness > 185),
        255,
        0
    ).astype(np.uint8)

    vertical_margin = int(mask.shape[0] * 0.04)
    horizontal_margin = int(mask.shape[1] * 0.04)

    mask[:vertical_margin, :] = 0
    mask[-vertical_margin:, :] = 0
    mask[:, :horizontal_margin] = 0
    mask[:, -horizontal_margin:] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    cleaned_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    number_of_labels, labels, statistics, centroids = (
        cv2.connectedComponentsWithStats(
            cleaned_mask,
            connectivity=8
        )
    )

    annotated_image = image.copy()
    object_count = 0

    image_area = image.shape[0] * image.shape[1]
    minimum_area = image_area * 0.015
    minimum_height = image.shape[0] * 0.20

    for label_number in range(1, number_of_labels):
        x = statistics[
            label_number,
            cv2.CC_STAT_LEFT
        ]
        y = statistics[
            label_number,
            cv2.CC_STAT_TOP
        ]
        width = statistics[
            label_number,
            cv2.CC_STAT_WIDTH
        ]
        height = statistics[
            label_number,
            cv2.CC_STAT_HEIGHT
        ]
        area = statistics[
            label_number,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_area and height >= minimum_height:
            object_count += 1

            cv2.rectangle(
                annotated_image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                3
            )

            cv2.putText(
                annotated_image,
                f"Object {object_count}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    print(f"Object count: {object_count}")

    return annotated_image


def main():
    parser = argparse.ArgumentParser(
        description="CSC 6302 image-processing toolbox"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input image"
    )

    parser.add_argument(
        "--op",
        required=True,
        choices=[
            "pick",
            "denoise",
            "sharpen",
            "edges",
            "count"
        ],
        help="Image-processing operation"
    )

    parser.add_argument(
        "--ksize",
        type=int,
        default=5,
        help="Odd kernel size"
    )

    parser.add_argument(
        "--amount",
        type=float,
        default=1.0,
        help="Unsharp-masking amount"
    )

    parser.add_argument(
        "--output",
        default="out.png",
        help="Output image path"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    if args.op == "pick":
        result = colour_picker(image)

    elif args.op == "denoise":
        result = denoise_image(
            image,
            args.ksize
        )

    elif args.op == "sharpen":
        result = sharpen_image(
            image,
            args.ksize,
            args.amount
        )

    elif args.op == "edges":
        result = detect_edges(image)

    elif args.op == "count":
        result = count_objects(
            image,
            args.ksize
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_path),
        result
    )

    if not success:
        raise OSError(
            f"Could not write output: {output_path}"
        )

    print(f"Operation completed: {args.op}")
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()