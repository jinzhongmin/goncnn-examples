# yolov5 example

## model convert
``` sh

#step1. git yolov5 src
git clone https://github.com/ultralytics/yolov5 
cd yolov5

#step2. export model, yolov5s.torchscript will be created in this folder
python export.py --weights yolov5s.pt --include torchscript

#step3.
# download pnnx from https://github.com/pnnx/pnnx/releases
# unzip pnnx

# files
# pnnx
#   |-- pnnx
#   |-- yolov5s.torchscript  // copy from step2
cd pnnx


#step4. convert model
pnnx yolov5s.torchscript inputshape=[1,3,640,640]f32

# get yolov5s.ncnn.bin, yolov5s.ncnn.param 
```