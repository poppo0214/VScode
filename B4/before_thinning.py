import vtk
import os
from vtkmodules.util import numpy_support
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import numpy as np
import glob
from utils.labeling import labeling



def extract(filter_array, segment_array):
    #labeling
    labeled_filter_array, num = labeling(filter_array)
    #labeled_segment_array, __ = labeling(segment_array)

    compoundarray = np.zeros_like(filter_array)

    for N in range(1, num+1):
        extracted_image = (labeled_filter_array==N)
        if np.sum(extracted_image*segment_array) != 0:
            compoundarray += extracted_image
    
    return(compoundarray)

if __name__ == '__main__':
    """共通分岐点を数える前に、正解データと同じvoxelに体積を持つ領域だけ抽出する"""
    filtered_path = r"d:\takahashi_k\commom_node\frg(1,17,2)_bn(20)"
    out_path = r'd:\takahashi_k\commom_node\frangi(1,17,2)_bn(20)_extracted'
    segmented_path = r'd:\takahashi_k\database\us\kobayashi\annotation'
    rootlist = glob.glob(f'{filtered_path}/*.vti')
    for root in rootlist:
            filtered_array, spa, ori = vtk_data_loader(root)
            filename = root.replace(filtered_path, "")
            filename = filename.replace(".vti", "")
            num = filename.find('Origin')
            personname = filename[0:num]

            print(personname, num)

            segmented_root = os.path.join(segmented_path + personname + "seg.vti")
            print(segmented_root)
            segmented_array, spa, ori = vtk_data_loader(segmented_root)

            extracted_array = extract(filtered_array, segmented_array)
            Out_root = os.path.join(out_path + filename + "_extracted.vti")
            print(Out_root)
            output = numpy_to_vtk(extracted_array, spa, ori)
            save_vtk(output, Out_root)
