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
