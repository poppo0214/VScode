import os
import pydicom

print("Enter the path where the dicom data exists")
path = input('>')
print("Enter the filename of dicom")
filename = input('>')
In_Dcm = os.path.join(path + "\\" + filename)
print("save dicom file data as txt file? y or n")
txt = input('>')

data = pydicom.dcmread(In_Dcm)
print(data)

if txt == "y":
    filename = filename.replace(".dcm", "")
    txtname = os.path.join(path + "\\" +filename + ".txt")
    f = open(txtname, 'a')
    f.write(str(data))
    f.close
    #D:\\takahashi_k\\database\\us\\peer\\annotation\\Kaho_0085