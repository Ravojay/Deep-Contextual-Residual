# Deep-Contextual-Residual
A self implementation of work: DEEP CONTEXTUAL RESIDUAL NETWORK FOR ELECTRON MICROSCOPY IMAGE SEGMENTATION IN CONNECTOMICS
usage: download ISBI EM dataset and make the train images and label images in seperate folders called: "images" and "labels" put these two folders in one folder called "data". Since the original dataset only contains 30 images, we nned to create more for training.

1.put all files in the same folder that contains "data".

2.run gen_data_v2 to add more images based on rotation, shifting and elastic transformation

3.run create_test_file to keep the original 30 images as test set

4.run mdoel_pixel_weight to train the model. During training, weight of epochs which has either the highest validation accuracy or the lowest validation loss will be saved.

5.run tester to generate result which takes the original 30 images as input

Evaluation metrics are available using fiji. Vrand and Vinfo in fiji takes either probability or binary images as input, so call binarify_test_file to create binary version of the original labels.
