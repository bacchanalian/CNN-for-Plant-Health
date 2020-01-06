# CNN-for-Plant-Health
CNN built from scratch for disease recognition and detection for use in agricultural monitoring setting.

Train, test and validation data all generated from the CrowdAI dataset, see manuscript.

Classification improvements require use of dropouts, undersampling privileged training sets and oversampling others. Larger epochs and GPU acceleration would be useful.

Test accuracy is 98% and validation accuracy is 92% however these results are not robust to detection. The major factor is training on the image dataset allows for differentiating between the categories (different kinds of leaves) but almost always returns false positives if region proposals contain something that isn't a classified leaf. To overcome this problem one must take advantage of transfer learning or simply run segmentation software that excludes all non-leaf elements.
