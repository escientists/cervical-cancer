# 11 04 2017
We tried the cervix segmentation kernel (https://www.kaggle.com/elenaran/intel-mobileodt-cervical-cancer-screening/cervix-segmentation-gmm/editnb). It performs ok. It crops the image to the relevant part. We could make a function that does all what we did in this notebook with a parameter for 'aggressiveness'. The kernel size for dilating and eroding would be a good thing to vary here.

We also tried out Carlos' code for superpixel segmentation (https://www.kaggle.com/elenaran/intel-mobileodt-cervical-cancer-screening/fork-of-cervix-segmentation-gmm/editnb). Note that you should use different parameters for grayscale than for rgb (See https://github.com/scikit-image/scikit-image/issues/1745). 

So we think we should first crop, then do the superpixel segmentation. After that, calculate some features, maybe cluster the superpixels on those features. Then bin and give to some classifier to classify on type [1,2,3].

We should also ask Jason about machines. We would like access to some easier machine to work with. 

25 04 2017

The search for a bumpiness measure. 

An initial idea for a bumpiness measure was to use entropy (see http://scikit-image.org/docs/dev/auto_examples/filters/plot_entropy.html). However, entropy depends on the number of colors and their distribution, and does not take into account if colors are similar. 

Another idea was to use a discrete fourier to see tell bumpiness using some frequency information. A nice intro about DFT on images is at https://www.cs.unm.edu/~brayer/vision/fourier.html However, I do not really see how we can use this information to extract a bumpiness measure.
