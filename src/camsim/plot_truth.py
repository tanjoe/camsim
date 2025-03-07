import os
import cv2
import json
import matplotlib.pyplot as plt
import typer
from pathlib import Path

def selectImageFromDir(img_dir: Path) -> Path:
    if not img_dir.exists():
        print(f"Error: '{img_dir}' directory not found.")
        raise typer.Exit(code=1)

    image_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    if not image_files:
        print(f"Error: No .png images found in the '{img_dir}' directory.")
        raise typer.Exit(code=1)

    print("Available images:")
    for i, file in enumerate(image_files):
        print(f"{i + 1}. {file}")

    while True:
        try:
            choice = int(input("Enter the number of the image to process: ")) - 1
            if 0 <= choice < len(image_files):
                image_filename = image_files[choice]
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    image_path = img_dir / image_filename
    return image_path


def draw_crosses_on_image(image_path: str, truth_json_path: str):
    """Loads an image, draws circles based on JSON data, and displays it using matplotlib."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(truth_json_path, "r") as f:
        truth_data = json.load(f)

    fig, ax = plt.subplots()
    # ax.imshow(image, interpolation="nearest")
    ax.imshow(image)

    cross_size = 2
    truth_cross_color = (0, 1, 0)
    for i, center in enumerate(truth_data):
        x, y = float(center[0]), float(center[1])
        # Draw a cross for the truth position
        ax.plot(
            [x - cross_size, x + cross_size],
            [y, y],
            color=truth_cross_color,
            linewidth=1,
        )
        ax.plot(
            [x, x],
            [y - cross_size, y + cross_size],
            color=truth_cross_color,
            linewidth=1,
        )
        ax.text(
            x,
            y - 5,
            str(i),
            color="yellow",
            fontsize=8,
            ha="center",
            va="center",
        )

    ax.set_axis_off()
    plt.show()


def main():
    """Draws circles on a selected image from the 'output' directory based on its corresponding JSON data."""
    output_dir = Path("output")
    image_path = selectImageFromDir(output_dir)
    truth_json_path = output_dir / (image_path.stem + "-loc_truth.json")

    if not image_path.exists():
        print(f"Error: Image file '{image_path}' not found.")
        raise typer.Exit(code=1)

    if not truth_json_path.exists():
        print(f"Error: Corresponding JSON file '{truth_json_path}' not found.")
        raise typer.Exit(code=1)

    draw_crosses_on_image(str(image_path), str(truth_json_path))


if __name__ == "__main__":
    typer.run(main)
