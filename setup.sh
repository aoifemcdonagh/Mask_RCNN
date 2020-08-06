# docker image: docker pull floydhub/tensorflow:1.14-gpu.cuda9cudnn7-py3_aws.44
# run docker image: docker run --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --runtime=nvidia --gpus all -v /home/gwy-dnn/field_project:/field_project --rm -it floydhub/tensorflow:1.14-gpu.cuda9cudnn7-py3_aws.44 bash

# Clone following repos in the same dir
# https://github.com/aoifemcdonagh/coco (checkout aoife_dev)
# https://github.com/aoifemcdonagh/Mask_RCNN (checkout aoife_dev)
# If required to preprocess LPIS data:
# https://github.com/aoifemcdonagh/InstanceSegmentation_Sentinel2 (checkout jupyter_cmd)

# run this setup.sh file in Mask_RCNN root dir

apt update
# required for opencv
apt install -y libsm6 libxext6 libxrender-dev
apt -y install python3-pip
apt install nano git python3-tk

pip3 install shapely==1.6.4
pip3 install geopandas==0.8.0
pip3 install matplotlib rasterio tqdm descartes scikit-image pathlib pprint

# uninstall keras from docker image before installing keras==2.1.5 in requirements.txt
pip uninstall keras

# requirements for Mask_RCNN
pip3 install -r requirements.txt

# required for building pycocotools
pip3 install imgaug Cython
# go to coco/PythonAPI
# make sure 'python' is changed to 'python3' in Makefile
cd ../coco/PythonAPI
make
make install
python3 setup.py install

