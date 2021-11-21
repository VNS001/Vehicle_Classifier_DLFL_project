# Vehicle_Classifier_DLFL_project
DLFL_mini_Project_image_classification


The .py file provided above is the code for making the tensorflow image classification model for classifying road Vehicles , the model is having accuracy around 88-95%


Transfer Learning is used to get the MobileNetV2 model trained on imageNet dataset for our project, and hence some of the top layers were locked / made Non-trainable
in this project , in order to retain their feature information.


The dataset required for our project was downloaded using the "simple_image_download" library.


Once required images for bike, bus, car, truck, scooty we downloaded they were splitted into train,validation,test datasets using "split-folders" library.
Both the above libraries can be downloaded from anaconda prompt using "pip install <--library name-->" command.
Comments are secified in the code whereever required for better understanding of what's happening.


Once the model is trained , validated and tested , it can be converted / saved into its .h5 and .tflite models with respective instructions specidied in the code file


Once the tflite model is generated , a label.txt is also made by simply writing the classes names in a noted pad .txt file , with each class name on new line
and also in the order in which they are fed into the model via training.


As imageDataGenerator was used to accessing and labeling our custom datasets , the classes were therefore arranged in alphabetical order as:

  1. bike
  2. bus
  3. car
  4. scooty
  5. truck
  
# Installation

## Android:
 
Finally After getting the .tflite and labels.txt files , one can download the android application code from here:
https://github.com/NSTiwari/Cartoon-Classification-on-Android-using-TF-Lite

and load/copy the .tflite model and labels.txt files in the assets folder and do the required changes as specified in above project link.

# App Screenshots are also mentioned above

One can download our app's apk file from the given drive link:

https://drive.google.com/file/d/1N24pVLpM5dRtb81BGd094FfrmvCpgI64/view?usp=sharing

NOTE: 
Allow installation from unknown sources.
Run the app.


Happy Learning !


