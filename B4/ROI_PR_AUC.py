import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.utils_vtk import vtk_data_loader

processed_path = "d:\\takahashi_k\\afterprocessing\\closing\\vessel_model\\add_noise\\"
processed_name = "vessel_v2_rev_pad_us_noise_shadow_frg(1)_close(5)_bn(5)_vp(100)"
gt_path = "d:\\takahashi_k\\afterprocessing\\closing\\vessel_model\\add_noise\\"
gt_name = "vessel_v2_rev_pad"
out_path = processed_path


processed_array, spa, ori = vtk_data_loader(os.path.join(processed_path + processed_name + ".vti"))
processed_array = processed_array/np.amax(processed_array)
processed_array = processed_array.flatten()
print(processed_array.shape, processed_array.dtype, np.amax(processed_array), np.amin(processed_array))
gt_array, spa, ori = vtk_data_loader(os.path.join(gt_path + gt_name + ".vti"))
gt_array = gt_array.astype(int)
gt_array = gt_array.flatten()
print(gt_array.shape, gt_array.dtype, np.amax(gt_array), np.amin(gt_array))

#ROC曲線
fpr, tpr, thresholds = metrics.roc_curve(gt_array, processed_array)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.plot(np.linspace(1, 0, len(fpr)), np.linspace(1, 0, len(fpr)), label='Random ROC curve (area = %.2f)'%0.5, linestyle = '--', color = 'gray')

plt.legend()
plt.title(f'ROC curve / AUC = {auc}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.savefig(os.path.join(out_path + processed_name + "_ROC.png"))
plt.clf()

precision, recall, thresholds = metrics.precision_recall_curve(gt_array, processed_array)

auc = metrics.auc(recall, precision)

plt.plot(recall, precision, label='PR curve (area = %.2f)'%auc)
plt.legend()
plt.title(f'PR curve / AUC = {auc}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.savefig(os.path.join(out_path + processed_name + "_PR.png"))