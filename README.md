﻿# Pytorch Federated Learning using Flower

 In this example, a simple convolutional neural network (CNN) is trained on the popular CIFAR-10 dataset using federated learning. CIFAR-10 can be used to train image classifiers that distinguish between images from ten different classes: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', and 'truck'.


# Instructions 
The server and client devices need to be connected to the same network. These devices can be PCs, and even System-on-Chip devices like the Raspberry Pi 4. You can also use the same computer as the server and clients, provided the computer has enough resources to do that. The minimum number of required clients is two for the Federated Learning to occur. This minimum number can be modified in the ```server.py``` code. Additionally in the ```server.py``` code, you can decide whether to save global models and aggregated results after every round.

1) Start by cloning the repository on the device(s) that will run as clients and the server.
2) Install dependencies on the device running as the server with the command below in a virutal environment or using anaconda:
```
pip install -r requirements_server.txt
```

3) Install dependencies on the device(s) running as client(s) with the command below in a virutal environment or using anaconda.
```
pip install -r requirements_client.txt
```

4) Update ```server_address``` value in both [server.py](server.py) and [client.py](client.py) with the IP address of the device running as the server. If you get an error message from ```server.py``` that says ```_ERROR_MESSAGE_PORT_BINDING_FAILED```, change the server's port to another one that is available.

5) Start the server by running [server.py](server.py) on the device that will act as the server:
```
python server.py
```

6) Start clients individually by running [client.py](client.py) on device that is running as a client. We need to pass the ```client_number``` so that we can load the dataset for that client. Currently the data is partitioned into two as NUM_CLIENTS = 2 but this can be changed by changing variable NUM_CLIENTS accordingly. The client_number can range from 1 to NUM_CLIENTS set. For example, to use a device as the first client, use --client_number=1:
```
python client.py --client_number=<CLIENT_NR>
```
