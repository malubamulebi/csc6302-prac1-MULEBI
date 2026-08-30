import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_psnr(clean, restored):
    """Calculate PSNR from its definition using NumPy."""
    clean_float = clean.astype(np.float64)
    restored_float = restored.astype(np.float64)

    mse = np.mean((clean_float - restored_float) ** 2)

    if mse == 0:
        return float("inf")

    return 10 * np.log10((255.0 ** 2) / mse)


def add_gaussian_noise(clean, sigma, random_generator):
    """Add Gaussian noise with the specified standard deviation."""
    noise = random_generator.normal(
        loc=0.0,
        scale=sigma,
        size=clean.shape
    )

    noisy_image = clean.astype(np.float64) + noise

    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(clean, proportion, random_generator):
    """Replace a proportion of pixels with black or white values."""
    noisy_image = clean.copy()

    selected_pixels = random_generator.random(clean.shape) < proportion
    salt_pixels = random_generator.random(clean.shape) < 0.5

    noisy_image[selected_pixels & salt_pixels] = 255
    noisy_image[selected_pixels & ~salt_pixels] = 0

    return noisy_image


def apply_filter(image, filter_name, kernel_size):
    """Apply one of the three required smoothing filters."""
    if filter_name == "Box":
        return cv2.blur(
            image,
            (kernel_size, kernel_size)
        )

    if filter_name == "Gaussian":
        return cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            sigmaX=0
        )

    if filter_name == "Median":
        return cv2.medianBlur(
            image,
            kernel_size
        )

    raise ValueError(f"Unknown filter: {filter_name}")


def save_comparison(noise_name, noisy_image, restored_images,
                    clean_image, output_directory):
    """Save a comparison figure for one type of noise."""
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))

    filter_names = ["Box", "Gaussian", "Median"]
    kernel_sizes = [3, 5, 7]

    for row, filter_name in enumerate(filter_names):
        axes[row, 0].imshow(
            noisy_image,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[row, 0].set_title(f"{noise_name} noise")

        for column, kernel_size in enumerate(kernel_sizes, start=1):
            restored = restored_images[
                (noise_name, filter_name, kernel_size)
            ]

            score = calculate_psnr(clean_image, restored)

            axes[row, column].imshow(
                restored,
                cmap="gray",
                vmin=0,
                vmax=255
            )
            axes[row, column].set_title(
                f"{filter_name} {kernel_size}x{kernel_size}\n"
                f"{score:.2f} dB"
            )

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(f"Filtering Results: {noise_name} Noise")
    figure.tight_layout()

    filename = (
        noise_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    figure.savefig(
        output_directory / f"{filename}_filter_comparison.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Task 2: noise and smoothing experiment"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/task2")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    clean_image = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE
    )

    if clean_image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    # A fixed seed makes the generated noise reproducible.
    random_generator = np.random.default_rng(2025025124)

    noisy_images = {
        "Gaussian": add_gaussian_noise(
            clean_image,
            sigma=15,
            random_generator=random_generator
        ),
        "Salt-and-pepper": add_salt_and_pepper_noise(
            clean_image,
            proportion=0.05,
            random_generator=random_generator
        )
    }

    cv2.imwrite(
        str(output_directory / "clean_grayscale.png"),
        clean_image
    )
    cv2.imwrite(
        str(output_directory / "gaussian_noise_sigma15.png"),
        noisy_images["Gaussian"]
    )
    cv2.imwrite(
        str(output_directory / "salt_pepper_noise_5percent.png"),
        noisy_images["Salt-and-pepper"]
    )

    results_table = []
    restored_images = {}

    for noise_name, noisy_image in noisy_images.items():
        for filter_name in ["Box", "Gaussian", "Median"]:
            for kernel_size in [3, 5, 7]:
                restored = apply_filter(
                    noisy_image,
                    filter_name,
                    kernel_size
                )

                score = calculate_psnr(
                    clean_image,
                    restored
                )

                restored_images[
                    (noise_name, filter_name, kernel_size)
                ] = restored

                results_table.append({
                    "Noise type": noise_name,
                    "Filter": filter_name,
                    "Kernel size": kernel_size,
                    "PSNR (dB)": round(score, 3)
                })

    csv_path = output_directory / "psnr_results.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=results_table[0].keys()
        )
        writer.writeheader()
        writer.writerows(results_table)

    for noise_name, noisy_image in noisy_images.items():
        save_comparison(
            noise_name,
            noisy_image,
            restored_images,
            clean_image,
            output_directory
        )

    print("\nPSNR RESULTS")
    print("-" * 65)

    for row in results_table:
        print(
            f"{row['Noise type']:<18}"
            f"{row['Filter']:<12}"
            f"Kernel {row['Kernel size']}  "
            f"{row['PSNR (dB)']:.3f} dB"
        )

    print("-" * 65)
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    main()