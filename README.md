# FedML_ObjectDetection
Based on FedML(https://github.com/FedML-AI/FedML) to develop object detection algorithms and test on our dataset.

# Create Docker Images
Download fedmltest:v1.tar from: https://drive.google.com/file/d/118jmEeA8rQf3BoP70xdeiFVDjvQUYz-k/view?usp=sharing.

Create a docker image by

    cat fedmltest:v1.tar | docker import - fedmltest:v1


After creating a docker image, run:

    docker run --shm-size 16G --gpus all -v ~/FedML_ObjectDetection:/media/FedML -it fedmltest:v1
    
Do the following instructions in the docker environment.

# Download COCO Dataset and Pretrained Weights
Go to fed_yolo/data, run

    bash get_coco_dataset.sh
    
Then, go to fed_yolo/weights, run
    
    bash download_weights.sh

# Run fed_yolo on COCO
In directory fed_yolo, run:
    
    python3 main.py --data_config config/coco.data  --pretrained_weights weights/darknet53.conv.74 --batch_size 4 --client_num_in_total 5 --client_num_per_round 5 --comm_round 100
