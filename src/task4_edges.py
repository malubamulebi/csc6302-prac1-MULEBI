import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def convolution_2d(image, kernel):
    """Apply a 2-D kernel using a NumPy implementation."""
    image_float = image.astype(np.float64)

    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(
        image_float,
        (
            (pad_height, pad_height),
            (pad_width, pad_width)
        ),
        mode="reflect"
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        padded_image,
        (kernel_height, kernel_width)
    )

    return np.einsum(
        "ijkl,kl->ij",
        windows,
        kernel
    )


def my_sobel(gray_image):
    """Calculate Sobel gradient magnitude using NumPy."""
    sobel_x_kernel = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        dtype=np.float64
    )

    sobel_y_kernel = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        dtype=np.float64
    )

    gradient_x = convolution_2d(
        gray_image,
        sobel_x_kernel
    )

    gradient_y = convolution_2d(
        gray_image,
        sobel_y_kernel
    )

    gradient_magnitude = np.sqrt(
        gradient_x ** 2 + gradient_y ** 2
    )

    return gradient_magnitude


def normalise_for_display(image):
    """Convert a floating-point result to a visible uint8 image."""
    return cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Task 4: edge detection"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/task4")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    gray_image = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE
    )

    if gray_image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    # Resize large images to make the from-scratch calculation faster.
    maximum_dimension = 700

    if max(gray_image.shape) > maximum_dimension:
        scale = maximum_dimension / max(gray_image.shape)

        gray_image = cv2.resize(
            gray_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    # Part 1: own Sobel implementation
    own_sobel = my_sobel(gray_image)

    opencv_gradient_x = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
        borderType=cv2.BORDER_REFLECT_101
    )

    opencv_gradient_y = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
        borderType=cv2.BORDER_REFLECT_101
    )

    opencv_sobel = np.sqrt(
        opencv_gradient_x ** 2
        + opencv_gradient_y ** 2
    )

    maximum_difference = np.max(
        np.abs(own_sobel - opencv_sobel)
    )

    own_sobel_display = normalise_for_display(own_sobel)
    opencv_sobel_display = normalise_for_display(opencv_sobel)

    cv2.imwrite(
        str(output_directory / "own_sobel.png"),
        own_sobel_display
    )

    cv2.imwrite(
        str(output_directory / "opencv_sobel.png"),
        opencv_sobel_display
    )

    sobel_figure, sobel_axes = plt.subplots(
        1,
        3,
        figsize=(12, 5)
    )

    sobel_axes[0].imshow(gray_image, cmap="gray")
    sobel_axes[0].set_title("Grayscale Image")

    sobel_axes[1].imshow(
        own_sobel_display,
        cmap="gray"
    )
    sobel_axes[1].set_title("Own NumPy Sobel")

    sobel_axes[2].imshow(
        opencv_sobel_display,
        cmap="gray"
    )
    sobel_axes[2].set_title("OpenCV Sobel")

    for axis in sobel_axes:
        axis.axis("off")

    sobel_figure.suptitle(
        f"Sobel Verification: Maximum Difference = "
        f"{maximum_difference:.6f}"
    )
    sobel_figure.tight_layout()

    sobel_figure.savefig(
        output_directory / "sobel_comparison.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(sobel_figure)

    # Part 2: Canny threshold experiment
    threshold_pairs = [
        (50, 100),
        (100, 200),
        (150, 250)
    ]

    canny_results = []

    for low_threshold, high_threshold in threshold_pairs:
        edge_image = cv2.Canny(
            gray_image,
            low_threshold,
            high_threshold
        )

        canny_results.append(
            (
                low_threshold,
                high_threshold,
                edge_image
            )
        )

        cv2.imwrite(
            str(
                output_directory
                / f"canny_{low_threshold}_{high_threshold}.png"
            ),
            edge_image
        )

    canny_figure, canny_axes = plt.subplots(
        1,
        4,
        figsize=(16, 5)
    )

    canny_axes[0].imshow(gray_image, cmap="gray")
    canny_axes[0].set_title("Grayscale Image")

    for index, result in enumerate(
        canny_results,
        start=1
    ):
        low_threshold, high_threshold, edge_image = result

        canny_axes[index].imshow(
            edge_image,
            cmap="gray"
        )
        canny_axes[index].set_title(
            f"Canny ({low_threshold}, {high_threshold})"
        )

    for axis in canny_axes:
        axis.axis("off")

    canny_figure.suptitle(
        "Effect of Canny Threshold Pairs"
    )
    canny_figure.tight_layout()

    canny_figure.savefig(
        output_directory / "canny_comparison.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(canny_figure)

    # Part 3: Laplacian of Gaussian
    gaussian_blurred = cv2.GaussianBlur(
        gray_image,
        (5, 5),
        sigmaX=1.0
    )

    laplacian = cv2.Laplacian(
        gaussian_blurred,
        cv2.CV_64F,
        ksize=3
    )

    log_display = normalise_for_display(
        np.abs(laplacian)
    )

    cv2.imwrite(
        str(output_directory / "laplacian_of_gaussian.png"),
        log_display
    )

    log_figure, log_axes = plt.subplots(
        1,
        3,
        figsize=(12, 5)
    )

    log_axes[0].imshow(gray_image, cmap="gray")
    log_axes[0].set_title("Grayscale Image")

    log_axes[1].imshow(
        own_sobel_display,
        cmap="gray"
    )
    log_axes[1].set_title("Sobel Magnitude")

    log_axes[2].imshow(
        log_display,
        cmap="gray"
    )
    log_axes[2].set_title("Laplacian of Gaussian")

    for axis in log_axes:
        axis.axis("off")

    log_figure.suptitle("Sobel and LoG Edge Character")
    log_figure.tight_layout()

    log_figure.savefig(
        output_directory / "sobel_log_comparison.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(log_figure)

    print("Task 4 completed successfully.")
    print(
        "Maximum absolute Sobel difference:",
        f"{maximum_difference:.6f}"
    )
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    main()