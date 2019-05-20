# NucleiSegmentation
Following are the instructions to test our state-of-the art deep learning based nuclei segmentation software using an AWS EC2 instance-

Step 1- Create an instance and configure it for using CUDA enabled Torch (refer to https://drive.google.com/file/d/0ByERBiBsEbuTUS0wdWQ2NUZGTm8/view)

Step 2- Get the software from Github

git clone https://github.com/neerajkumarvaid/NucleiSegmentation
---
Step 3- Test our state-of-the art nuclei segmentation model

cd NucleiSegmentation
---
th predict_full_mask.lua
---
Results will be saved in the /data/testing-data/40x/results folder

Step 4- Zip the results folder and download in your laptop's "Downloads" folder

zip  -r results.zip results
---
scp –i key.pem user@ip:~/NucleiSegmentation/data/testing-data/40x/results.zip  ~/Downloads/
---
This will give you three images from CNN output  1.png, 2.png and 3.png. Please refer to nucleisegmentationbenchmark.weebly.com for post-processing and model details.
