import os
import pydicom

print("Enter the path where the dicom data exists")
path = input('>')
print("Enter the filename of dicom")
filename = input('>')
In_Dcm = os.path.join(path + "\\" + filename)
#print(In_dcm)

data = pydicom.dcmread(In_Dcm)
spacing = data[0x200d, 0x3303].value
spacing = str(spacing)
spacing = spacing.lstrip("['")
spacing = spacing.rstrip("']")
spacing = spacing.split("', '")
for i in range(len(spacing)):
    spacing[i] = float(spacing[i])

print("spacing:", spacing)
print("Enter the Maximum diameter of the blood vessel to be considered[mm]")
diameter = input('>')
diameter = float(diameter)
max_sgm = diameter/(2*min(spacing))
min_sgm = diameter/(2*max(spacing))
print("Max sigma: ", max_sgm, "\nmin sigma: ", min_sgm)