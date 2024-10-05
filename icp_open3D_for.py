import open3d as o3d
import numpy as np
import copy
import os
import glob
import time

"""https://whitewell.sakura.ne.jp/Open3D/ICPRegistration.html"""
"""https://www.open3d.org/docs/latest/tutorial/t_pipelines/t_icp_registration.html"""
"""https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html"""


### Helper visualization function
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def pcd_mtx(pcd_array, mtx):
    paded_array = np.pad(pcd_array,((0,0),(0,1)),  mode="constant", constant_values=1)
    moved_list = []
    for i in range(paded_array.shape[0]):
        origin_point = paded_array[i, :].T
        moved_point = np.dot(mtx, origin_point)
        moved_list.append(moved_point)
        #print(origin_point, moved_point)
    moved_array = np.asarray(moved_list)[:, :3]
    return moved_array



if __name__ == "__main__":
    #for theta in range(0, 360, 30):
    for error in range(0, 16, 1):
        target_path = r"D:\takahashi_k\registration(model)\vessel(filled_2)"
        #source_path =  os.path.join(target_path + f"\\rotation_{theta}deg")
        source_path =  os.path.join(target_path + f"\\translation_{error}mm")
        out_path = os.path.join(source_path + f"\\icp")
        #out_path = r"D:\takahashi_k\reg/istration\translation2\icp2"
        matrix_path = os.path.join(out_path + f"\\TrsMtx_EM")
        #matrix_path = r"D:\takahashi_k\registration\translation2\icp2\matrix2"
        output_regied_source = True
        treg = o3d.pipelines.registration

        if not os.path.exists(out_path):
            #print(out_path)
            os.mkdir(out_path)

        if not os.path.exists(matrix_path):
            os.mkdir(matrix_path)
        

        #各種設定
        # 最大対応距離: ターゲット点群内の対応点を近傍探索するときの，ソース点群内の各点からの距離の半径[mm]
        max_correspondence_distance = 10000
        # 初期同時変換行列(shape [4, 4] of type Float64 on CPU:0 device)
        init_source_to_target = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])
        # ICP法の決定
        estimation = treg.TransformationEstimationPointToPoint()
        # 収束条件
        #fitnessはオーバーラップエリア（inlier対応数/ターゲット内のポイント数）を測定する。数値が大きいほどほど良い。
        #inlier_rmseはすべてのinlier対応の(Root Mean Squared Error, 平均二乗誤差平方根)を測定する。
        #relative_rmse:対応点間の距離、max_iteration: 最大イテレーション回数
        criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
                                            relative_rmse=0.000001,
                                            max_iteration=10000)
        #criteria = treg.ICPConvergenceCriteria(max_iteration=100000)
        # Down-sampling voxel-size. If voxel_size < 0, original scale is used.
        voxel_size = -1
        # # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
        # save_loss_log = True
        # # Example callback_after_iteration lambda function:
        # callback_after_iteration = lambda updated_result_dict : print("Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
        # updated_result_dict["iteration_index"].item(),
        # updated_result_dict["fitness"].item(),
        # updated_result_dict["inlier_rmse"].item()))

        target_rootlist = glob.glob(f'{target_path}/*.pcd')
        for target_root in target_rootlist:
            print(target_root)
            target = o3d.io.read_point_cloud(target_root)
            target.estimate_normals()
            #対応するソースのpcdを見つける
            target_name = target_root.replace(target_path, "")
            target_name = target_name.replace(".pcd", "")
            target_name = target_name.replace("\\", "")
            print(target_name)
            
            #print(f'{source_path}\\{target_name}*.pcd')
            print(source_path)
            source_rootlist = glob.glob(f'{source_path}/{target_name}*.pcd')
            for source_root in source_rootlist:
                print(source_root)
                source = o3d.io.read_point_cloud(source_root)
                source_array = np.asarray(source.points)
                source.estimate_normals()
                source_name = source_root.replace(source_path, "") 
                source_name = source_name.replace(".pcd", "")
                source_name = source_name.replace("\\", "")
                print(source_name)

                s = time.time()
                reg_p2p = treg.registration_icp(source, target, max_correspondence_distance,
                                            init_source_to_target, estimation, criteria)

                icp_time = time.time() - s
                #print("Time taken by ICP: ", icp_time)
                #print(reg_p2p.transformation)
                # print("Inlier Fitness: ", reg_p2p.fitness)
                # print("Inlier RMSE: ", reg_p2p.inlier_rmse)

                #draw_registration_result(source, target, reg_p2p.transformation)
                if output_regied_source:
                    out_pcd = os.path.join(out_path + "\\" + source_name + "_icped_to_" + target_name + f".pcd")
                    # regied_source = source.transform(reg_p2p.transformation)
                    regied_array = pcd_mtx(source_array, reg_p2p.transformation)
                    regied_source = o3d.geometry.PointCloud()
                    regied_source.points = o3d.utility.Vector3dVector(regied_array)
                    o3d.io.write_point_cloud(out_pcd, regied_source)
                

                matrix_csv = os.path.join(matrix_path + "\\" + source_name + "_icped_to_" + target_name + f".tsv")           
                np.savetxt(matrix_csv, reg_p2p.transformation, delimiter='\t')
                

            