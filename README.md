# FedML_ObjectDetection
Based on FedML(https://github.com/FedML-AI/FedML) to develop object detection algorithms and test on our dataset.

# Create Docker images

# Download COCO dataset

# Run fed_yolo on COCO
in directory fed_yolo, run:
python3 main.py --data_config config/coco.data  --pretrained_weights weights/darknet53.conv.74 --batch_size 4 --client_num_in_total 5 --client_num_per_round 5 --comm_round 100
