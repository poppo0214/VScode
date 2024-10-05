from skimage.morphology import skeletonize_3d
from skimage.morphology import skeletonize
import numpy as np
import glob
import os
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
from utils.labeling import labeling
import numpy as np

"""入力は2値画像"""


if __name__ == '__main__':
    print("loading vti...")
    In_path = r"d:\takahashi_k\commom_node\7_AS_or_masked_VP(500)_closing(5)_VP(5000)_extracted"
    Out_path = os.path.join(In_path)
    #Out_path = os.path.join(In_path + "_thinning")
    os.makedirs(Out_path, exist_ok=True)

    root_list = glob.glob(f'{In_path}/*.vti')
    for root in root_list:
        array, spa, ori = vtk_data_loader(root)

        #centerline = skeletonize_3d(array)     #skeletonize_3dとkeletonizeの出力結果は同じ
        centerline = skeletonize(array)

        # (x, y, z) = array.shape
        # for i in range(1, z-1, 1):
        #     for j in range(1, y-1, 1):
        #         for k in range(1, x-1, 1):
        #             count = 0
        #             for a in range(-1, 2, 1):
        #                 for b in range(-1, 2, 1):
        #                     for c in range(-1, 2, 1):
        #                         count += centerline[k+c, j+b, i+a]
        #             if count == 4:
        #                 junction[k, j, i] = 1

        print("saving as vti...")
        # array to image
        filename = root.replace(In_path, "")
        filename = filename.replace(".vti", "")
        centerline = numpy_to_vtk(centerline, spa, ori)
        # save output file(.vti)
        Out_Img = os.path.join(Out_path + filename + "_thinning.vti")
        save_vtk(centerline, Out_Img)

        # # array to image
        # junction = numpy_to_vtk(junction, spa, ori)
        # # save output file(.vti)
        # Out_Img = os.path.join(outpath + filename + "_junction.vti")
        # save_vtk(junction, Out_Img)

