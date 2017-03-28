# cervical-cancer
Repo for https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening 

# TODO:

* Transform to grayscale
* Histogram equalization
* a) Try out shallow network on grayscale
  * downscale to 512x512
* b) Find bumpiness/entropy measure
  * segment on texture
    * explore methods in opencv
  * visualize entropy-image
  * feed entropy image into NN
  * Use segmentation + statistical methods to classify

#Some links for rings detection in images:

http://stackoverflow.com/questions/28407487/how-to-calculate-radius-of-ring-in-matlab
http://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile 
