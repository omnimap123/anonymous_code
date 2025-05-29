import numpy as np
import open3d as o3d

def draw_camera(transform, line_color=(0., 1.0, 0.), plane_color=(0.0, 1.0, 0.0), scale=0.05):
    points = np.array([
        [0, 0, 0],  # 相机中心
        [-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2],  # 近裁剪面
    ]) * scale

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  
        [1, 2], [2, 3], [3, 4], [4, 1],  
    ]

    points = (transform @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :3]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([line_color] * len(lines)) 
    
    plane = o3d.geometry.TriangleMesh()
    plane_vertices = np.array([
        points[1], points[2], points[3], points[4] 
    ])
    plane.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2],  # 三角形 1
        [0, 2, 3]   # 三角形 2
    ])
    plane.vertices = o3d.utility.Vector3dVector(plane_vertices)
    plane.paint_uniform_color(plane_color)  # 设置矩形平面的颜色为红色
    plane.compute_vertex_normals()  # 计算法线，确保正确渲染
    return line_set, plane



def create_camera_trajectory_line():
    trajectory_line_set = o3d.geometry.LineSet()
    trajectory_line_set.points = o3d.utility.Vector3dVector([])  # 初始为空
    trajectory_line_set.lines = o3d.utility.Vector2iVector([])   # 初始为空
    trajectory_line_set.colors = o3d.utility.Vector3dVector([])  # 初始为空
    return trajectory_line_set


def update_camera_trajectory(trajectory_line_set, points):
    if len(trajectory_line_set.points) > 0:
        current_points = np.array(trajectory_line_set.points)
        trajectory_line_set.points = o3d.utility.Vector3dVector(np.vstack([current_points, points]))  # 添加新的点
        lines = [[i, i + 1] for i in range(len(current_points)-1, len(current_points) + len(points) - 1)]  # 新增连接线
        colors = [[1, 0, 0] for _ in lines]  # 轨迹线为红色
        if len(trajectory_line_set.lines) > 0:
            trajectory_line_set.lines = o3d.utility.Vector2iVector(np.vstack([trajectory_line_set.lines, lines]))
            trajectory_line_set.colors = o3d.utility.Vector3dVector(np.vstack([trajectory_line_set.colors, colors]))
        else:
            trajectory_line_set.lines = o3d.utility.Vector2iVector(lines)
            trajectory_line_set.colors = o3d.utility.Vector3dVector(colors)
    else:
        trajectory_line_set.points = o3d.utility.Vector3dVector(points)  # 添加新的点
    return trajectory_line_set