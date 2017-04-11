# 11 04 2017
We tried the cervix segmentation kernel (https://www.kaggle.com/elenaran/intel-mobileodt-cervical-cancer-screening/cervix-segmentation-gmm/editnb). It performs ok. It crops the image to the relevant part. We could make a function that does all what we did in this notebook with a parameter for 'aggressiveness'. The kernel size for dilating and eroding would be a good thing to vary here.

We also tried out Carlos' code for superpixel segmentation (https://www.kaggle.com/elenaran/intel-mobileodt-cervical-cancer-screening/fork-of-cervix-segmentation-gmm/editnb). Note that you should use different parameters for grayscale than for rgb (See https://github.com/scikit-image/scikit-image/issues/1745). 

So we think we should first crop, then do the superpixel segmentation. After that, calculate some features, maybe cluster the superpixels on those features. Then bin and give to some classifier to classify on type [1,2,3].

We should also ask Jason about machines. We would like access to some easier machine to work with. 
