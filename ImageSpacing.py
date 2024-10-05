import cv2
import numpy as np
import vtk

class Philips_iU22_screenshot_png2VTI():
  def __init__(self,filename):
    self.originSize=[1050,1680]
    self.center=[745,207]
    self.axes=[600,600]
    self.color=[1]
    self.offsetAngle=40
    self.startAngle=0
    self.endAngle=100
    self.imageROI=[1024,1024]
    self.scaleRoi=[1115,235,1127,800]
    self.vtkOrigin=(0,0,0)
    self.vtkSpacing=(0,0,0)
    if isinstance(filename,str):
       self.image=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    else:
       raise ValueError('Unsupported argument type.')
  
  
  def generate_mask(self):
     mask = np.zeros(self.originSize, dtype=np.uint8)
     cv2.ellipse(mask,self.center,self.axes,self.offsetAngle,self.startAngle,self.endAngle,255,-1,cv2.LINE_AA)
     self.Mask=mask
  
  def generate_cropped_image(self):
     self.generate_mask()
     self.cropped_image=cv2.bitwise_and(self.image,self.image,mask=self.Mask)
  
  #この関数で、iU22の画像のリサイズができるので、これを使ってリサイズされたpngを取得してください
  def resize_Mask_R_CNN_image(self):
     self.generate_cropped_image()
     dx=self.imageROI[1]/2-self.center[0]
     dy=-self.center[1]
     M=np.float32([[1,0,dx],[0,1,dy]])   
     shifted_image=cv2.warpAffine(self.cropped_image,M,(self.imageROI))
     self.resizeimage=cv2.resize(shifted_image,(512,512))
  
  def getResize_Mask_R_CNN_Image(self,resizeFileName):
     self.resize_Mask_R_CNN_image()
     cv2.imwrite(resizeFileName,self.resizeimage)

  def scale_image(self):
     self.scale_image=self.image[self.scaleRoi[1]:self.scaleRoi[3],self.scaleRoi[0]:self.scaleRoi[2]]
  
  def get_VTK_information(self):
     self.scale_image()
     point=(0,0)
     while (self.scale_image[point[1]-1, point[0]-1])<100:
        if point[0]<self.scale_image.shape[1]:
           point=(point[0]+1,point[1])
        elif point[1] < self.scale_image.shape[0]:
           point = (0, point[1] + 1)
        else:
           print("Can not find Scale Area")
           exit(1)
     scale_points=[]
     x=point[0]
     y=point[1]+3
     scale_count=0
     while (self.scale_image[y, x+3])<250:
        scale_points.append(self.scale_image[y,x])
        if scale_points[len(scale_points)-1]-scale_points[len(scale_points)-2]>200:
           scale_count+=1
        y+=1
     
     scale_length=50.0 if scale_count==4 else 10.0
     scale=scale_length/(len(scale_points)+4)
     resized_scale=scale*2
     self.vtkSpacing=[resized_scale,resized_scale,resized_scale]
     self.vtkOrigin=[-resized_scale*self.imageROI[0]/4,0,0]
     self.scale=scale_length
  
  def png2VTK(self):
    self.resize_Mask_R_CNN_image()
    self.get_VTK_information()
    self.vtkImage=vtk.vtkImageData()
    self.vtkImage.SetDimensions(self.resizeimage.shape[1],self.resizeimage.shape[0],1)
    self.vtkImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,1)
    self.vtkImage.SetOrigin(self.vtkOrigin)
    self.vtkImage.SetSpacing(self.vtkSpacing)
    vtk_array=self.vtkImage.GetPointData().GetScalars()
    vtk_array.SetVoidArray(self.resizeimage,np.prod(self.resizeimage.shape),1)
  
  #リサイズ前の元画像を入れて、この関数を使うと、スペーシング、原点をそろえ、リサイズされたvtiファイルが出てきます
  def get_VTK(self,vtk_filename):
     self.png2VTK()
     writer=vtk.vtkXMLImageDataWriter()
     writer.SetFileName(vtk_filename)
     writer.SetInputData(self.vtkImage)
     writer.Write()
  
   #この関数は、マスク画像をvtkに変える方法です。
   #spacing,originは、getVTKで得た画像から取得してください
  def get_Mask_VTK(self,vtk_filename,mask_filename):
     reader=vtk.vtkXMLImageDataReader()
     reader.SetFileName(vtk_filename)
     reader.Update()

     vtk_image=reader.GetOutput()
     spa=vtk_image.GetSpacing()
     ori=vtk_image.GetOrigin()
     dim=vtk_image.GetDimensions()
     vtk_mask=vtk.vtkImageData()
     vtk_mask.SetDimensions(dim)
     vtk_mask.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,1)
     vtk_mask.SetOrigin(ori)
     vtk_mask.SetSpacing(spa)
     vtk_array=vtk_mask.GetPointData().GetScalars()
     vtk_array.SetVoidArray(self.image,np.prod(self.image.shape),1)

     writer=vtk.vtkXMLImageDataWriter()
     writer.SetFileName(mask_filename)
     writer.SetInputData(vtk_mask)
     writer.Write()




obj=Philips_iU22_screenshot_png2VTI("Image1.png")
obj.get_Mask_VTK("output.vti","mask.vti")
# obj.get_VTK("output.vti")
# obj.getResize_Mask_R_CNN_Image("resized.png")