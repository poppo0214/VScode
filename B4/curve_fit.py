import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import lmfit as lf
import lmfit.models as lfm
#from lmfit.models import PolynomialModel
import os
import csv

if __name__ == '__main__':
    r_list = ["1(0.5-1.5)", "2(1.5-2.5)", "3(2.5-3.5)", "4(3.5-4.5)", "5(4.5-6)", "7(6-8)", "9(8-10)", "11(10-12.5)", "14(12.5-15.5)", "17(15.5-18.5)", "20(18.5-22.5)", "25(22.5-27.5)", "30(27.5-32.5)"]
    for r in r_list:
        inpath = "d:\\takahashi_k\\model\\analyze\\all"
        outpath = inpath     
        #r = '_1(0.5-1.5)'
        inroot = os.path.join(inpath + '\\' + r + f"\\allmodel_{r}_points.csv")
        outtxt = os.path.join(inpath + '\\' + r + f"\\fitting.txt")
        outcsv = os.path.join(inpath + '\\' + r + f"\\fitting_evaulation.csv")
        # inroot = os.path.join(inpath + r + f"_points.csv")
        # outtxt = os.path.join(inpath + r + f"_fitting.txt")
        # outcsv = os.path.join(inpath + r + f"_fitting_evaulation.csv")
        
        deg = 1                     #多項式フィッティングの次数

        data = []
        # x = []
        # y = []
        with open(inroot) as fin:
            reader = csv.reader(fin,quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                data.append([row[0], row[3]])
                # x.append(row[0])
                # y.append(row[3])
        fin.close()

        #データをソートする
        data = np.array(data)
        sort_order = np.argsort(data[:, 0])
        sorted_data = data[sort_order]
        x = sorted_data[:, 0]
        y = sorted_data[:, 1]

        ftxt = open(outtxt, 'a')
        fcsv = open(outcsv, 'a')
        fcsv.write(r)
        fcsv.write("\n次数,カイ自乗,AIC,BIC,R2,係数\n")
        for deg in range(1,8):
            model = lfm.PolynomialModel(degree=deg)
            params = model.guess(y, x=x)
            result = model.fit(y, params, x=x)
            
            ftxt.write(f"deg = {deg}")
            ftxt.write(result.fit_report())
            ftxt.write("\n\n")

            fcsv.write(f"{deg},{result.chisqr},{result.aic},{result.bic},{result.rsquared},{result.best_values}\n")

            outfunc = os.path.join(inpath + '\\' + r + f"\\fitting{deg}.png")     
            #outfunc = os.path.join(inpath + r + f"fitting{deg}.png")         
            fig, ax = plt.subplots(dpi=130)
            result.plot_fit(ax=ax)
            plt.savefig(outfunc,dpi=130)
        ftxt.close()
        fcsv.close()
