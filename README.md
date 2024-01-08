# GIT_disease_Detection
Detection of Gastrointestinal Disease Using Deep Learning Strategies

This research aims to utilize deep learning models to classify endoscopic images into two categories: those with gastrointestinal disease present and those with gastrointestinal disease absent. Additionally, several image preprocessing techniques such as channelization and segmentation will be employed.

**Data Acquisition and Processing**
The HyperKvasir Dataset consists of endoscopic images of both healthy and diseased gastrointestinal tracts that were publicly available. The dataset contains a total of 3000 images, categorized into two groups. The first group includes normal gastrointestinal tract images without any disease, while the second group consists of abnormal gastrointestinal tract images with diseases such as polyps, ulcerative colitis, and others. Each group has 1500 images. To train the convolutional neural network, the images were resized to 300 x 300, while for the Transfer learning network (AlexNet), the images were resized to 227 x 227.

**Image Channelization**
Image channelization is a technique used in image processing to split an image into multiple channels, each of which represents a specific color component or feature of the image. In this research, the images are separated into their respective colour channels: Red, Green, and Blue, as well as the grayscale channel.

**Experimental Results**
Experimental results from the Blue, Green, and Red colour channels and the original image were run through AlexNet and a CNN model. 

![image](https://github.com/Honolulu69/GIT_disease_Detection/assets/54552155/6c1b2735-d6e3-4ed5-86dd-321b586c6fb8)
![image](https://github.com/Honolulu69/GIT_disease_Detection/assets/54552155/8034d965-d13c-4dc7-91a8-8eb7a20b7670)
![Screenshot 2023-04-18 140504](https://github.com/Honolulu69/GIT_disease_Detection/assets/54552155/fb8f331f-defc-476a-a16e-9730780c3dad)

Publication incoming**
