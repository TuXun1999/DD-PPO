import genesis as gs
from PIL import Image

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

cam = scene.add_camera(
    res    = (640, 480),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = True,
)

cam2 = scene.add_camera(
    res    = (640, 480),
    pos    = (0.0, 0.0, 4.0),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = True,
)

scene.build()

# 渲染rgb、深度、分割掩码和法线图
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=False, normal=False)

# cam.start_recording()
import numpy as np

for i in range(200):
    scene.step()
    rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=False, normal=False)
    # cam.render()

    rgb2, depth2, segmentation2, normal2 = cam2.render(rgb=True, depth=True, segmentation=False, normal=False)
    # cam2.render()

    # 保存 RGB 图像
    # rgb_image = Image.fromarray((rgb).astype(np.uint8))  # 如果 rgb 范围在 [0, 1]，则需要乘以 255
    # rgb_image.save(f'./RGB/rgb_frame_{i:03d}.png')  # 使用 3 位格式化保存
print(type(rgb))
print(type(rgb2))
print(rgb.shape)
print(rgb2.shape)
# cam.stop_recording(save_to_filename='video1112.mp4', fps=60)
# # 打印 RGB 图像的信息
# print("RGB Information:")
# print(f"Shape: {rgb.shape}")       # 打印形状
# print(f"Type: {type(rgb)}")       # 打印类型
# print(f"Data Type: {rgb.dtype}")  # 打印数据类型

# # 打印深度图的信息
# print("\nDepth Information:")
# print(f"Shape: {depth.shape}")       # 打印形状
# print(f"Type: {type(depth)}")       # 打印类型
# print(f"Data Type: {depth.dtype}")  # 打印数据类型