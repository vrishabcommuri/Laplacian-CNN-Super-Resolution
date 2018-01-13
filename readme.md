# Laplacian CNN Super Resolution

CMU 15-112 Term Project

### Background 
Applications such as Snapchat and Facebook provide streamlined platforms for sending images between users. However, due to the high volume of images transferred between users on these platforms, photos sent through these applications are either downsampled or captured at a lower resolution than the resolution of a user's camera. This facilitates lower transfer times for the sender, but results in inferior-quality photos for the recipient.

### Aim
To address the limitations presented above, the use of a Laplacian Pyramid Super Resolution Network (LapSRN) is proposed to quickly upsample a downsampled or natively low-resolution facial image for a receiving user. Allowing images to be transferred at an extremely low resolution will reduce the cost of data transfer and improve latency for users.

### Methodology
The methods that will be utilized to complete this project are outlined in this section.
#### Components
The components for this project, inluding libraries, hardware, and the LapSRN network are listed below. All code will be implemented in Python.

1. A LapSRN coded in the Pytorch deep learning library
2. A webcam interface through the OpenCV library
2. A graphichal user interface coded in tkinter 

#### Methods
The first step of this project was to create a working LapSRN in Pytorch. The discriminator in the network uses the Leaky ReLU activation function as it mitigates the effects of ReLUs dying when a large gradient is passed through them. A pretrained network (sourced from the LapSRN paper code) was further trained on facial images. The training data for the network was the CelebFaces Attributes Dataset (CelebA). The images were split into training and testing sets, downsampled, and fed into the network. The output was compared to the ground truth images. The image upscaling process is illustrated in the figure below. 

***

<p align="center"> 
<img src="https://github.com/vrishabcommuri/Laplacian-CNN-Super-Resolution/blob/master/samples/upsampling_process.png">
</p>

The image upscaling process carried out by the LapSRN. An input is fed into the rightmost level of the pyramid. The process of extracting nonlinear features and then upsampling via transposed convolutional layers is repeated at each level of the pyramid. Though this diagram shows three such layers, the adaptation of the LapSRN implemented in this project only has two levels, yielding an upsampled image 4x upscaled as opposed to 16x upscaled in a network with three layers.

***

After the scripts for interfacing with the LapSRN were complete, an application was created to allow the user to easily edit and save downsampled images and upsample the downsampled images. The application utilizes opencv and tkinter to interface with the user's webcam and create the user interface respectively.

### Results

Using this network, images of 512 x 512 pixels were downsampled using bicubic interpolation to 128 x 128 pixels, saved, and upsampled back to 512 x 512 pixels. The results presented with good preservation of quality and a compression ratio of 16:1 (512x512/128x128)! Here is a side-by-side comparison of an input and output pair:

<p align="center"> 
<img src="https://github.com/vrishabcommuri/Laplacian-CNN-Super-Resolution/blob/master/samples/result_1.gif">
</p>

The input sizes can be any square size as well. As such, the input size can be slowly increased creating some very nice animations of the outputs. These outputs have input sizes ranging from 32 x 32 to 256 x 256 pixels and output sizes ranging from 128 x 128 to 1024 x 1024 pixels. 

<p align="center"> 
<img src="https://github.com/vrishabcommuri/Laplacian-CNN-Super-Resolution/blob/master/samples/result_2.gif">
</p>
<p align="center"> 
<img src="https://github.com/vrishabcommuri/Laplacian-CNN-Super-Resolution/blob/master/samples/result_3.gif">
</p>

If you liked these animations, here is a fun video I made detailing the project's UI and showing some more of the network's outputs. Since the output image sequences are not constrained to the gif file format, they look a lot better in the video as well. 

[Enjoy!](https://youtu.be/lE5S5sHzwXE)

### TODO

* Enable interface between different machines. 
* Enable the user to send color images

