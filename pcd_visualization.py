import open3d as o3d
import numpy as np
import os

"""青→１，オレンジ→２"""

# path1 = r"D:\takahashi_k\registration(model)\forUSE\target"
# filename1 = "Y-target(filled)"
path1 = r"D:\takahashi_k\registration(differentBVN)\YAMAGUCHI\forABE"
filename1 = "IM_0328(filled)"
root1 = os.path.join(path1 + "\\" + filename1 + ".pcd")
pcd_data1 = o3d.io.read_point_cloud(root1)
pcd_data1.paint_uniform_color([0, 0.651, 0.929])
#o3d.visualization.draw_geometries([pcd_data1])


path2 = r"D:\takahashi_k\registration(differentBVN)\YAMAGUCHI\forABE\RANSAC+ICP(surface)"
filename2 = "IM_0330(surface)_Regied_to_IM_0328(surface)"
# path2 = r"D:\takahashi_k\registration(differentBVN)\yamaguchi1\source"
# filename2 = "IM_0329"
root2 = os.path.join(path2 + "\\" + filename2 + ".pcd")
pcd_data2 = o3d.io.read_point_cloud(root2)
pcd_data2.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([pcd_data1, pcd_data2])

# path3 = r"D:\takahashi_k\registration\test\rotation_120deg\verification2"
# filename3 = "tube(filled)_120-0_removed"
# root3 = os.path.join(path3 + "\\" + filename3 + ".pcd")
# pcd_data3 = o3d.io.read_point_cloud(root3)

# o3d.visualization.draw_geometries([pcd_data1, pcd_data2, pcd_data3])