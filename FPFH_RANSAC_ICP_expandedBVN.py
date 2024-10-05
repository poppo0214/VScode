import open3d as o3d
import numpy as np
import copy
import os
import glob
import time
"""https://whitewell.sakura.ne.jp/Open3D/GlobalRegistration#Extract-geometric-feature"""


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # radius_normal = voxel_size * 2
    radius_normal = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 10
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    #radius_feature = voxel_size * 5
    radius_feature = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 10
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_root, target_root, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    # source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    # target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    source = o3d.io.read_point_cloud(source_root)
    target = o3d.io.read_point_cloud(target_root)
    trans_init = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #distance_threshold = 1000
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target):
    distance_threshold = 10000
    init_source_to_target = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    # result = o3d.pipelines.registration.registration_icp(
    #     source, target, distance_threshold, result_ransac.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001,
                                            relative_rmse=0.000001,
                                            max_iteration=10000)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold,
                                            init_source_to_target, estimation, criteria)

    return result

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
    path =  r"D:\takahashi_k\registration(expandedBVN)\makino"
    target_name = "makino-target"
    target_root = os.path.join(path + "\\" + target_name + ".pcd")
        
    for error in range(10, 81, 10):
        for theta in range(0, 91, 10):
            source_path =  os.path.join(path + f"\\source\\{error}mm-{theta}deg")
            dir_path = os.path.join(source_path + f"\\FPFH_RANSAC_icp")
            TrsMtx_path = os.path.join(dir_path + f"\\TrsMtx_EM")
            RansacMtx_path = os.path.join(dir_path + f"\\RansacMtx_EM")
            IcpMtx_path = os.path.join(dir_path + f"\\IcpMtx_EM")
            output_regied_source = True
            treg = o3d.pipelines.registration

            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            if not os.path.exists(TrsMtx_path):
                os.mkdir(TrsMtx_path)
            if not os.path.exists(RansacMtx_path):
                os.mkdir(RansacMtx_path)
            if not os.path.exists(IcpMtx_path):
                os.mkdir(IcpMtx_path)
            
    
            source_rootlist = glob.glob(f'{source_path}/*.pcd')
            for source_root in source_rootlist:
                print(source_root)
                # source = o3d.io.read_point_cloud(source_root)
                # source_array = np.asarray(source.points)
                # source.estimate_normals()
                source_name = source_root.replace(source_path, "") 
                source_name = source_name.replace(".pcd", "")
                source_name = source_name.replace("\\", "")
                print(source_name)

                """FPFHによる特徴点抽出DownSampling"""
                #s = time.time()
                voxel_size = 0.005  # means 5mm for the dataset
                source, target, source_down, target_down, source_fpfh, target_fpfh = \
                        prepare_dataset(source_root, target_root, voxel_size)
                #draw_registration_result(source_down, target_down, [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

                """DownSamplingの結果を用いたRANSAC"""
                result_ransac = execute_global_registration(source_down, target_down,
                                                            source_fpfh, target_fpfh,
                                                            voxel_size)
                #print(result_ransac)
                RansacMtx_csv = os.path.join(RansacMtx_path + "\\" + source_name + "_RANSAC_to_" + target_name + f".tsv")           
                np.savetxt(RansacMtx_csv, result_ransac.transformation, delimiter='\t')
                #draw_registration_result(source_down, target_down, result_ransac.transformation)
                source_array = np.asarray(source.points)
                ransac_array = pcd_mtx(source_array, result_ransac.transformation)
                ransac = o3d.geometry.PointCloud()
                ransac.points = o3d.utility.Vector3dVector(ransac_array)


                """RANSACの結果を踏まえたICP（DownSampingしない元データ）"""
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

                result_icp = refine_registration(ransac, target)
                IcpMtx_csv = os.path.join(IcpMtx_path + "\\" + source_name + "_icped_to_" + target_name + f".tsv")           
                np.savetxt(IcpMtx_csv, result_icp.transformation, delimiter='\t')
                print(result_icp)
                #draw_registration_result(source, target, reg_p2p.transformation)
                if output_regied_source:
                    out_pcd = os.path.join(dir_path + "\\" + source_name + "_Regied_to_" + target_name + f".pcd")
                    regied_array = pcd_mtx(ransac_array, result_icp.transformation)
                    regied = o3d.geometry.PointCloud()
                    regied.points = o3d.utility.Vector3dVector(regied_array)
                    o3d.io.write_point_cloud(out_pcd, regied)

                                    
                """トータルの同時変換行列の算出、書き出し"""
                TrsMtx = np.dot(result_icp.transformation, result_ransac.transformation)
                print(TrsMtx)
                TrsMtx_csv = os.path.join(TrsMtx_path + "\\" + source_name + "_Regied_to_" + target_name + f".tsv")           
                np.savetxt(TrsMtx_csv, TrsMtx, delimiter='\t')
