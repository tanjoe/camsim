import os
import datetime
import cv2
import pyrender
import trimesh
import json
import numpy as np
import pyglet
import matplotlib.pyplot as plt


def projectWorldToImage(
    camera: pyrender.IntrinsicsCamera, camera_pose: np.ndarray, world_point: np.ndarray
) -> tuple[float, float]:
    # Transform the point to camera coordinates
    point_3d_camera = np.linalg.inv(camera_pose) @ world_point
    point_3d_camera = point_3d_camera[:3]
    X, Y, Z = point_3d_camera

    # Project the 3D point to 2D image coordinates
    x = camera.cx + (camera.fx * X) / -Z
    # Flip the Y-axis to follow OpenCV convention
    y = camera.cy - (camera.fy * Y) / -Z

    return (x, y)


def draw_crosses_on_image(image_path: str, truth_json_path: str):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(truth_json_path, "r") as f:
        truth_data = json.load(f)

    fig, ax = plt.subplots()
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

    plt.show()


class MyViewer(pyrender.Viewer):
    def __init__(
        self,
        scene: pyrender.Scene,
        camera: pyrender.IntrinsicsCamera,
        interested_points: list[np.ndarray],
        viewport_size: tuple[int, int],
        render_flags=None,
        viewer_flags=None,
        registered_keys=None,
        run_in_thread=False,
        **kwargs,
    ):
        self.camera = camera
        self.interested_points = interested_points
        super().__init__(
            scene,
            viewport_size,
            render_flags,
            viewer_flags,
            registered_keys,
            run_in_thread,
            **kwargs,
        )

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ENTER:
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            pyglet.image.get_buffer_manager().get_color_buffer().save(
                f"output/{timestamp}.png"
            )

            # Get the current camera pose
            camera_pose = self.scene.get_pose(self.scene.main_camera_node)
            print(f"Camera pose: {camera_pose}")
            np.savetxt(f"output/{timestamp}-camera_pose.txt", camera_pose)

            loc_truth = []
            for point in self.interested_points:
                loc_truth.append(projectWorldToImage(self.camera, camera_pose, point))
            with open(f"output/{timestamp}-loc_truth.json", "w") as truth_file:
                json.dump(loc_truth, truth_file, indent=4)

            draw_crosses_on_image(
                f"output/{timestamp}.png", f"output/{timestamp}-loc_truth.json"
            )

        return super().on_key_press(symbol, modifiers)


def createBoard(image_path: str) -> pyrender.Mesh:
    # Create a texture from the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    texture = pyrender.Texture(
        source=image,
        source_channels="RGB",
        width=image.shape[1],
        height=image.shape[0],
    )

    # Create a material using the texture
    material = pyrender.MetallicRoughnessMaterial(
        baseColorTexture=texture,
        emissiveTexture=texture,
        emissiveFactor=[0.9, 0.9, 0.9],
        doubleSided=False,
        smooth=False,
    )

    # Create a plane mesh with UV coordinates
    # Compute vertices based on image resolution to keep the aspect ratio
    half_w = image.shape[1] / 1000.0 / 2.0
    half_h = image.shape[0] / 1000.0 / 2.0
    vertices = np.array(
        [
            [-half_w, -half_h, 0],  # Bottom-left
            [half_w, -half_h, 0],  # Bottom-right
            [-half_w, half_h, 0],  # Top-left
            [half_w, half_h, 0],  # Top-right
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            [0, 1, 2],  # First triangle
            [1, 3, 2],  # Second triangle
        ],
        dtype=np.uint32,
    )

    # Define UV coordinates for texture mapping
    uv_coords = np.array(
        [
            [0, 0],  # Bottom-left
            [1, 0],  # Bottom-right
            [0, 1],  # Top-left
            [1, 1],  # Top-right
        ],
        dtype=np.float64,
    )

    # Create a Trimesh object with vertices, faces, and UV coordinates
    plane = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv_coords, image=image),
    )

    # Create a Pyrender mesh from the Trimesh object
    plane_mesh = pyrender.Mesh.from_trimesh(plane, material=material, smooth=False)
    return plane_mesh


def computeCircleCoordinates(json_path: str, board_mesh: pyrender.Mesh) -> np.ndarray:
    coordinates: list[list[float]] = []
    with open(json_path, "r") as content:
        board_info = json.load(content)
        image_size = board_info["image_size"]
        centers = board_info["centers"]

        width_ratio = board_mesh.extents[0] / image_size[0]
        height_ratio = board_mesh.extents[1] / image_size[1]
        start_x = board_mesh.bounds[0][0]
        start_y = board_mesh.bounds[0][1]

        for c in centers:
            # When a circle is drawn with center at (1, 1) in OpenCV, its actual PIXEL DISTANCE to the left
            # corner is (1.5, 1.5). So take care of the 0.5 here
            u = c[0] + 0.5
            # Flip y to make it follows OpenGL convention (+Y should be upward)
            v = image_size[1] - c[1] - 0.5
            x = start_x + u * width_ratio
            y = start_y + v * height_ratio
            coordinates.append([x, y, 0])
    return np.array(coordinates)


def main() -> None:
    plane_mesh = createBoard("resource/board.png")
    plane_pose = np.eye(4)
    plane_pose[2, 3] = -10
    plane_node = pyrender.Node(mesh=plane_mesh, matrix=plane_pose)

    centers = computeCircleCoordinates("resource/board.json", plane_mesh)
    transformed_centers = []
    for center in centers:
        center_homo = np.append(center, 1)
        transformed_centers.append(plane_pose @ center_homo)

    # Create a camera at the origin looking down the z-axis
    view_width = 1920
    view_height = 1080
    camera = pyrender.IntrinsicsCamera(
        fx=view_width, fy=view_width, cx=(view_width / 2), cy=(view_height / 2)
    )
    camera_pose = np.eye(4)
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)

    # Create a scene
    scene = pyrender.Scene(bg_color=[0, 0, 0, 1])
    scene.add_node(plane_node)
    scene.add_node(camera_node)

    # Render the scene using the interactive viewer
    os.makedirs("./output", exist_ok=True)
    MyViewer(
        scene,
        camera=camera,
        interested_points=transformed_centers,
        viewport_size=(view_width, view_height),
    )


if __name__ == "__main__":
    main()
