import vtk
from vtkmodules.util import numpy_support
import numpy as np
from skimage.morphology import opening, closing, cube
import sys

def vtk_to_numpy(data):
  """
  This function is to transform vtk to numpy
  Args
      data: vtk data
  Return: numpy data
  """
  temp = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
  dims = data.GetDimensions()
  global spa
  spa = data.GetSpacing()
  component = data.GetNumberOfScalarComponents()
  if component == 1:
    numpy_data = temp.reshape(dims[2], dims[1], dims[0])
    numpy_data = numpy_data.transpose(2,1,0)
  elif component == 3 or component == 4:
    if dims[2] == 1: # a 2D RGB image
      numpy_data = temp.reshape(dims[1], dims[0], component)
      numpy_data = numpy_data.transpose(0, 1, 2)
      numpy_data = np.flipud(numpy_data)
    else:
      raise RuntimeError('unknow type')
  return numpy_data

def numpy_to_vtk(data, spa, ori, multi_component=False, type='char'):
  '''
  multi_components: rgb has 3 components
  type:float or char
  '''
  if type == 'float':
    data_type = vtk.VTK_FLOAT
  elif type == 'char':
    data_type = vtk.VTK_UNSIGNED_CHAR
  else:
    raise RuntimeError('unknown type')
  if multi_component == False:
    if len(data.shape) == 2:
      data = data[:, :, np.newaxis]
    flat_data_array = data.transpose(2,1,0).flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
    shape = data.shape
  else:
    assert len(data.shape) == 3, 'only test for 2D RGB'
    flat_data_array = data.transpose(1, 0, 2)
    flat_data_array = np.reshape(flat_data_array, newshape=[-1, data.shape[2]])
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
    shape = [data.shape[0], data.shape[1], 1]
  
  img = vtk.vtkImageData()
  img.GetPointData().SetScalars(vtk_data)
  img.SetDimensions(shape[0], shape[1], shape[2])
  img.SetSpacing(spa[0], spa[1], spa[2])
  img.SetOrigin(ori[0], ori[1], ori[2])
  return img

def save_vtk(img, output_path):
  writer = vtk.vtkXMLImageDataWriter()
  writer.SetFileName(output_path)
  writer.SetInputData(img)
  writer.Write()

def vtk_data_loader(data_path):
  """
  This function is to load vtk data
  Args
      data_path: vtk data path
  Return: vtk data transformed to numpy
  """
  vtk_reader = vtk.vtkXMLImageDataReader()
  vtk_reader.SetFileName(data_path)
  vtk_reader.Update()
  vtk_data = vtk_reader.GetOutput()
  
  spa = vtk_data.GetSpacing()
  ori = vtk_data.GetOrigin()

  npdata = vtk_to_numpy(vtk_data).astype(np.float32)
  
  # data = np.zeros((x, y, z))

  return (npdata, spa, ori)

def save_vtp(poly, output_path):
  writer = vtk.vtkXMLPolyDataWriter()
  writer.SetFileName(output_path)
  writer.SetInputData(poly)
  writer.Write()

def vtp_data_loader(data_path):
  reader=vtk.vtkXMLPolyDataReader()
  reader.SetFileName(data_path)
  reader.Update()
  vtp_data = reader.GetOutput()
  return(vtp_data)

def vti_to_vtp(vti_array, spacing, origin):
  # ポイントクラウドのための空のリストを作成
  points = vtk.vtkPoints()                   #プローブ座標系
  vertices = vtk.vtkCellArray()

  # 3次元バイナリデータをスキャンしてポイントクラウドを生成
  dimensions = vti_array.shape
  print(dimensions)
  for z in range(dimensions[2]):  # Z方向
    for y in range(dimensions[1]):  # Y方向
      for x in range(dimensions[0]):  # X方向
        # 各座標のバイナリ値を取得
        value = vti_array[x, y, z]
        # バイナリ値が1ならポイントを追加
        if value >= 1:
          #print(value)
          # 座標とスペーシングに基づいて点の位置を計算
          point_x = origin[0] + x * spacing[0]
          point_y = origin[1] + y * spacing[1]
          point_z = origin[2] + z * spacing[2]

          point = points.InsertNextPoint(point_x, point_y, point_z)
          vertices.InsertNextCell(1)
          vertices.InsertCellPoint(point)
          
          polydata = vtk.vtkPolyData()
          polydata.SetPoints(points)
          polydata.SetVerts(vertices)

  return(polydata)

def vtp_to_vti(polydata, spacing_mode, root=None, spacing=None):
  """
  spacing
    calculate:vtpの位置座標からspacingを計算.非推奨.
    default:[0.5, 0.5, 0.5]
    reference:他のvtiファイルのspacingを参照。rootに参照するvtiファイルのrootを渡すこと。
    manual:手動でspacingを指定。spacingにspacingを渡すこと。
  """
  #vtpからorigin（インデックスが0の点の座標）とboundsを取得
  bounds = polydata.GetBounds()                 #vtpの境界を取得
  origin = [bounds[0], bounds[2], bounds[4]]    #境界の端をoriginに指定
  num = polydata.GetNumberOfPoints()            #vtpのポイント数を取得

  if spacing_mode == "calcurate":        #polydataの位置座標の差の平均をspacingとする
    pre_point = np.array(polydata.GetPoint(0))
    diff_point = 0
    avenum = round(num/100)                  #spacingの計算に用いるpointの数:全ポイント数の1/100（整数）
    for i in range(1,avenum):
      point = np.array(polydata.GetPoint(i))
      diff_point += np.abs(point - pre_point)
      pre_point = point
    spacing = np.array(diff_point/avenum, dtype='float')

  elif spacing_mode == "default":
    spacing = [0.5, 0.5, 0.5]

  elif spacing_mode == "reference":
    vtk_reader = vtk.vtkXMLImageDataReader()
    vtk_reader.SetFileName(root)
    vtk_reader.Update()
    vtk_data = vtk_reader.GetOutput()
    spacing = vtk_data.GetSpacing()
    #print(spacing)
  
  elif spacing_mode == "manual":
    spacing = spacing
  
  else:
    print('Please select the appropriate mode')
    sys.exit()

  #print(f"converted vti origin:{origin}\nconverted vti spacing:{spacing}")
  #boundsとspacingからdimentionsを計算して、空の配列を生成
  x_dim = round((bounds[1]-bounds[0])/spacing[0]) + 1
  y_dim = round((bounds[3]-bounds[2])/spacing[1]) + 1
  z_dim = round((bounds[5]-bounds[4])/spacing[2]) + 1
  dimentions = (x_dim, y_dim, z_dim)
  #print("dimentions:", dimentions)
  img_array = np.zeros(dimentions)

  #polyデータの座標を取得して、近くの座標に対応する配列を1にする
  for i in range(num):
    point = np.array(polydata.GetPoint(i))
    index = []
    index.append(round((point[0]-bounds[0])/spacing[0]))
    index.append(round((point[1]-bounds[2])/spacing[1]))
    index.append(round((point[2]-bounds[4])/spacing[2]))
    img_array[index[0], index[1], index[2]] = 1
  
  if type(spacing) != list:
    spacing = list(spacing)

  kernel = round(1/min(spacing))
  img_array = closing(img_array, cube(kernel))         #穴埋め

  #print(img_array.shape)
  imagedata = numpy_to_vtk(img_array, spacing, origin)
  return(imagedata)

