import cv2
import json
import matplotlib.pyplot as plt


def main() -> None:
    image = cv2.imread("resource/board.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open("resource/board.json", "r") as f:
        board_info = json.load(f)
        centers = board_info["centers"]
        x_ratio = image.shape[1] / board_info["image_size"][0]
        y_ratio = image.shape[0] / board_info["image_size"][1]

    fig, ax = plt.subplots()
    ax.imshow(image)

    cross_size = 2
    cross_color = (0, 1, 0)
    for center in centers:
        x, y = float(center[0]) * x_ratio, float(center[1]) * y_ratio
        # Draw a cross for the truth position
        ax.plot(
            [x - cross_size, x + cross_size],
            [y, y],
            color=cross_color,
            linewidth=1,
        )
        ax.plot(
            [x, x], [y - cross_size, y + cross_size], color=cross_color, linewidth=1
        )

    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    main()
