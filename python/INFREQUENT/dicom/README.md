# DICOM

1. link
   * [wikipedia](https://en.wikipedia.org/wiki/DICOM)
   * [official site](http://dicom.nema.org/standard.html)
   * [DICOM standard introduction and Overview](http://dicom.nema.org/medical/dicom/current/output/chtml/part01/PS3.1.html)
   * DICOM standard-Information Object Definitions: see [link](http://dicom.nema.org/medical/dicom/2017b/output/chtml/part03/sect_C.7.6.2.html)
2. `pydicom` module: `conda install -n python3 -c clinicalgraphics pydicom`

## image plane module

1. image position and image orientation
   * IP (0020,0032), x, y, and z coordinates of the upper left hand corner of the image
2. Slice Location: the relative position of the image plane expressed in mm

## Get started with pydicom

see [link](http://pydicom.readthedocs.io/en/stable/getting_started.html)

```python
import dicom as dcm
ds = dcm.read_file("export0002.dcm")
ds.PatientName
ds.dir("setup") #get a list of tags with "setup" somewhere in the name
# ds.PatientSetupSequence[0] #missed
# ds.PatientSetupSequence[0].PatientPosition = "HFP" #missed
ds.save_as("tbd01.dcm")
```

## Pydicom User Guide

see [link](http://pydicom.readthedocs.io/en/stable/pydicom_user_guide.html)

1. base object: `dataset`
2. `ds.dir("pat")`
3. `ds.PatientName`
4. `ds[0x10,0x10].value`
5. DataElement
   * `tag`: a DICOM tag
   * `VR`: Vaulue Representation
   * `VM`: Value Multiplicity
   * `value`: the actual value

```python
# data_element
z1 = ds.data_element("PatientsName")
z1.VR, z1.value # ('PN', 'Last^First^mid^pre')

# check for the existence of a particular tag
"PatientsName" in ds # True

# remove a data element
del ds[tag]

# work with pixel data
z1 = ds.PixelData
z2 = ds.pixel_array # numpy required
```

## Working with Pixel Data

see [link](http://pydicom.readthedocs.io/en/stable/working_with_pixel_data.html)

1. functions or properties
   * `dcm.read_file()`
   * `ds.PixelData`
   * `ds.pixel_array`
   * `ds.PixelData = ds.pixel_array.tostring()`
   * `ds.save_as('newfilename.dcm')`
   * `ds.Rows`
   * `ds.Columns`

## Viewing Images

see [link](http://pydicom.readthedocs.io/en/stable/viewing_images.html)

1. DICOM viewer programs
2. pydicom with matplotlib
3. pydicom with Tkinter
4. pydicom with Python Imaging Library (PIL)
5. pydicom with wxPython

## [如何应用Python处理医学影像学中的DICOM信息](http://www.jianshu.com/p/df64088e9b6b)

1. python package
   * [pydicom](http://pydicom.readthedocs.io/en/stable/getting_started.html) 无法处理压缩像素图像（如JPEG），无法处理分帧动画图像
   * [SimpleITK](http://www.simpleitk.org/): Insight Segmentation and Registration Toolkit
   * PIL: Python Image Library
   * OpenCV

```python
# pre-processing
import SimpleITK as sitk
from PIL import Image
import pydicom as dcm
import numpy as np
import cv2

def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    return img_array, frame_num, width, height

def loadFileInformation(filename):
    information = {}
    ds = dicom.read_file(filename)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    information['NumberOfFrames'] = ds.NumberOfFrames
    return information

# 应用PIL来检查图像是否被提取
def showImage(img_array, frame_num = 0):
    img_bitmap = Image.fromarray(img_array[frame_num])
    return img_bitmap

# 采用CLAHE (Contrast Limited Adaptive Histogram Equalization)技术来优化图像
def limitedEqualize(img_array, limit = 4.0):
    img_array_list = []
    for img in img_array:
        clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = (8,8))
        img_array_list.append(clahe.apply(img))
    img_array_limited_equalized = np.array(img_array_list)
    return img_array_limited_equalized

def writeVideo(img_array):
    frame_num, width, height = img_array.shape
    filename_output = filename.split('.')[0] + '.avi'
    video = cv2.VideoWriter(filename_output, -1, 16, (width, height))
    for img in img_array:
        video.write(img)
    video.release()
```

## 常见医疗扫描图像处理步骤

see [link](http://shartoo.github.io/medical_image_process/)

1. [booz-allen-hamilton/DSB3Tutorial](https://github.com/booz-allen-hamilton/DSB3Tutorial)
2. file format
   * dcm
   * mhd: SimpleITK, `img_array`的数组不是直接的像素值，而是相对于CT扫描中原点位置的差值
3. software [mango](http://ric.uthscsa.edu/mango/index.html)
4. HU (Hounsfield Unit) 放射剂量（下表）
   * `rescale slope`一般等于`1`
   * `intercept`一般等于`-1024`
   * `HU = pixel_value*rescale_slope + intercept`
5. LUT (Look Up Table)：像素灰度值的映射表，将实际采样到的像素灰度值经过一定的变换如阈值、反转、二值化、对比度调整、线性变换等，变成了另外一 个与之对应的灰度值，这样可以起到突出图像的有用信息，增强图像的光对比度的作用
6. 去除超过 -2000的pixl_array，CT扫描边界之外的像素值固定为-2000(dicom和mhd都是这个值)。第一步是设定这些值为0，当前对应为空气（值为0）
7. 绘制HU的统计直方图（略）
8. 重新采样，映射到一个同构分辨率 $1mm \times 1mm \times 1mm$
9. 绘制肺部扫描的3D图像（略）
10. 输出一个病人扫描的切面图（略）
11. 定义分割出CT切面里面肺部组织的函数
    * `from skimage.segmentation import clear_border`
    * `from skimage.measure import label, regionprops`
    * `from skimage.morphology import disk, binary_erosion`

**substance**|**HU**
:-----:|:-----:
air|-1000
lung|-500
脂肪|(-100,-50)
water|0
CSF|15
kidney|30
blood|(30,45)
muscle|(10,40)
灰质|(37,45)
白质|(20,30)
liver|(40,60)
软组织, constrast|(-100,300)
bone|(-700,300)

```python
# read mhd file
import SimpleITK as sitk
itk_img = sitk.ReadImage(img_file)
img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
num_z, height, width = img_array.shape
origin = np.array(itk_img.GetOrigin()) # x,y,z  Origin in world coordinates (mm)
spacing = np.array(itk_img.GetSpacing()) #spacing of voxels in world coor. (mm)

HU = pixel_value * rescale_slope + rescale_intercept
```
