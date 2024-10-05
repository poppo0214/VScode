import os
import glob

if __name__ == '__main__':
    In_path = r"D:\takahashi_k\registration(model)\forUSE\source"   
    rootlist = glob.glob(f'{In_path}/*')

    for root in rootlist:
        #newroot = root.replace("_wall", "_nonwall")
        #newroot = root.replace("[", "_")
        newroot = root.replace(".", "(filled).")
        #newroot = root.replace("\\Kobayashi", "\\1_Kobayashi")
        os.rename(root, newroot)

# if __name__ == '__main__':
#     In_path = r"D:\takahashi_k\registration(expandedBVN)\makino\source" 
#     for num in range(0,1000, 1):
#         num = str(num).zfill(3)  
#         rootlist = glob.glob(f'{In_path}/*/*/*Few-{num}.tsv')

#         for root in rootlist:
            
#                 #newroot = root.replace("_wall", "_nonwall")
#                 #newroot = root.replace("[", "_")
#                 newroot = root.replace(f"Few-{num}.tsv", f"-Few{num}.tsv")
#                 #newroot = root.replace("\\Kobayashi", "\\1_Kobayashi")
#                 os.rename(root, newroot)

        