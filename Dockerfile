FROM floydhub/tensorflow:1.14-gpu.cuda9cudnn7-py3_aws.44
COPY . /tmp
RUN apt update; apt install -y git python3-pip
RUN apt install nvidia-cuda-dev

#RUN pip3 install shapely==1.6.4 matplotlib geopandas rasterio tqdm descartes scikit-image jupyter
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install jupyter

RUN sh ./setup.sh

ENV image sample.jpg
CMD python3 samples/field/inference_window.py -i image