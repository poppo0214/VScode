import pandas as pd
import seaborn as sns
import matplotlib
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
import os
import xlrd

#$$で囲んだ部分が斜体になる
path = r'D:\takahashi_k\TabFig\comperasion'
filename = 'frg_exclusive_Pre_Post'
#label_list = ["$bth$=10", "$bth$=20", "$bth$=30"]
#label_list = ["Frangi","PP1 + Frangi","Frangi", "PP1 + Frangi","Frangi", "PP1 + Frangi"]
# label_list = ["Frangi","PP2($t$=5) + Frangi","PP2($t$=10) + Frangi","PP2($t$=20) + Frangi",
#               "Frangi","PP2($t$=5) + Frangi","PP2($t$=10) + Frangi","PP2($t$=20) + Frangi",
#               "Frangi","PP2($t$=5) + Frangi","PP2($t$=10) + Frangi","PP2($t$=20) + Frangi",]
#label_list = ["F", "F\n+ RSLR($rth=1000$) + closing($ksize$=5)",
            #   "PP1 + F", "PP1 + F\n+RSLR($rth=1000$) + closing($ksize$=5)",
            #   "Exclusive", "Exclusive\n+ RSLR($rth=1000$) + closing($ksize$=5)",
            #   "PP1 + Exclusive", "PP1 + Exclusive\n+RSLR($rth=1000$) + closing($ksize$=5)"]
# label_list = ["VWmodel", "nonVWmodel", "Combined", "Exclusive",
#               "VWmodel", "nonVWmodel", "Combined", "Exclusive",
#               "VWmodel", "nonVWmodel", "Combined", "Exclusive",
#               "VWmodel", "nonVWmodel", "Combined", "Exclusive",
#               "VWmodel", "nonVWmodel", "Combined", "Exclusive",
#               "VWmodel", "nonVWmodel", "Combined", "Exclusive",]
# label_list = ["VWmodel", "nonVWmodel", "Combined", "Exclusive",
#               "VWmodel", "nonVWmodel", "Combined", "Exclusive",
#               "VWmodel", "nonVWmodel", "Combined", "Exclusive"]
# label_list = ["Combined", "PP1 + Combined", "Exclusive", "PP1 + Exclusive",
#               "Combined", "PP1 + Combined", "Exclusive", "PP1 + Exclusive",
#               "Combined", "PP1 + Combined", "Exclusive", "PP1 + Exclusive"]
# label_list = ["PP1+Frangi", "PP1+Frangi+RSLR($rth$=100)",  "PP1+Frangi+RSLR($rth$=250)","PP1+Frangi+RSLR($rth$=500)", 
#               "PP1+Frangi+RSLR($rth$=1000)", "PP1+Frangi+RSLR($rth$=2500)", "PP1+Frangi+RSLR($rth$=5000)"]
# label_list = ["PP1+Exclusive", "PP1+Exclusive\n+RSLR($rth$=100)", "PP1+Exclusive\n+RSLR($rth$=250)", "PP1+Exclusive\n+RSLR($rth$=500)", 
#               "PP1+Exclusive\n+RSLR($rth$=1000)", "PP1+Exclusive\n+RSLR($rth$=2500)", "PP1+Exclusive\n+RSLR($rth$=5000)"]
#label_list = ["PP1+Frangi", "PP1+Frangi+closing($ksize$=5)", "PP1+Frangi+closing($ksize$=10)", "PP1+Frangi+closing($ksize$=15)"]
#label_list = ["PP1+Exclusive", "PP1+Exclusive\n+closing($ksize$=5)", "PP1+Exclusive\n+closing($ksize$=10)", "PP1+Exclusive\n+closing($ksize$=15)"]
# label_list = ["PP1+Frangi\n+RSLR($rth$=500)", "PP1+Frangi\n+RSLR($rth$=500)+closing($ksize$=5)", 
#               "PP1+Frangi\n+RSLR($rth$=500)+closing($ksize$=10)", "PP1+Frangi\n+RSLR($rth$=500)+closing($ksize$=15)",]
# label_list = ["PP1+Exclusive\n+RSLR($rth$=500)", "PP1+Exclusive\n+RSLR($rth$=500)+closing($ksize$=5)", 
#               "PP1+Exclusive\n+RSLR($rth$=500)+closing($ksize$=10)", "PP1+Exclusive\n+RSLR($rth$=500)+closing($ksize$=15)",]
# label_list = ["PP1+Frangi+RSLR($rth$=500)\n+closing($ksize$=5)", "PP1+Frangi+RSLR($rth$=500)\n+closing($ksize$=5)+RSLR($rth$=1000)",
#               "PP1+Frangi+RSLR($rth$=500)\n+closing($ksize$=5)+RSLR($rth$=2500)","PP1+Frangi+RSLR($rth$=500)\n+closing($ksize$=5)+RSLR($rth$=5000)",
#               "PP1+Frangi+RSLR($rth$=500)\n+closing($ksize$=5)+RSLR($rth$=10000)","PP1+Frangi+RSLR($rth$=500)\n+closing($ksize$=5)+RSLR($rth$=25000)"]
# label_list = ["PP1+Exclusive+RSLR($rth$=500)\n+closing($ksize$=5)", "PP1+Exclusive+RSLR($rth$=500)\n+closing($ksize$=5)+RSLR($rth$=1000)",
#               "PP1+Exclusive+RSLR($rth$=500)\n+closing($ksize$=5)+RSLR($rth$=2500)","PP1+Exclusive+RSLR($rth$=500)\n+closing($ksize$=5)+RSLR($rth$=5000)",
#               "PP1+Exclusive+RSLR($rth$=500)\n+closing($ksize$=5)+RSLR($rth$=10000)","PP1+Exclusive+RSLR($rth$=500)\n+closing($ksize$=5)+RSLR($rth$=25000)"]
# label_list = ["Previous method", "Proposed method"]
#label_list = ["PP1+Frangi", "PP1+Frangi+Combined Post-Processing", "PP1+Exclusive", "PP1+Exclusive+Combined Post-Processing"]
label_list = ["Frangi", "PP1+Frangi+Combined Post-Processing", "Exclusive", "PP1+Exclusive+Combined Post-Processing"]



#index = "Precision"
index_list = ["Precision", "Recall", "Dice"]
for index in index_list:
    # エクセルファイルからデータを読み込む
    excel_root = os.path.join(path + "\\" + filename + '.xlsx')
    output_root = os.path.join(path + f'\\{filename}_{index}.png')
    print(output_root)

    df = pd.read_excel(excel_root, sheet_name=index, index_col=0)
    #label_list = df.columns.values
    value = df.values
    num = value.shape[1]
    print(num)
    data = []

    for i in range(num):
        list = value[:, i]
        list = list.tolist()
        data.append(list)
    print(data)


    #plt.rcParams['font.family'] = 'Times New Roman' 
    fig = plt.figure(figsize=(16, 8))  # プロットのサイズを設定
    ax = fig.add_subplot()
    ax.set_axisbelow(True)
    plt.grid(axis = 'y', linestyle = 'dotted')

    ax.boxplot(data, labels = label_list,
                patch_artist=True,  # 細かい設定をできるようにする
                showmeans = True,
                medianprops=dict(color='black', linewidth=1),  # 中央値の線の設定
                meanprops={"marker":"x", "markeredgecolor":"black"},
                whis=(0, 100), widths=0.5,
                boxprops=dict(facecolor='darkgrey')
            )

    plt.ylabel(index,fontsize=23)
    #plt.xlabel(DAI, fontsize=23)
    plt.xticks(fontsize=10)
    plt.ylim([0,1])
    plt.savefig(output_root)

