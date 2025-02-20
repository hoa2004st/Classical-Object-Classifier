# Classical-Object-Classifier
## Description:
This mini projects compares performance of multiple global feature extraction and classification methods on a pretrained dataset.
- Dataset: CIFAR-10 (60 000 images divided into 10 classes)
- Feature extraction: Hu's Moments, PHOG (pyramid histogram of oriented gradients) (for KNN and SVM)
- Classification methods: KNN (k-nearest neighbours), SVM (supported vector machine), VGG16, and YOLO11.
## Result:
After training on CIFAR-10 dataset, this is the result:
![result.png](result/result.png)
## Tuning Hyper Parameter and Training Process:
Tuning Hyper Parameter and Training Process are recorded as follow:
- Tuning of KNN and SVM models on data extracted using Hu's Moments:
![KNN_and_SVM_on_Hu's_Moments.png](result/KNN_and_SVM_on_Hu's_Moments.png)
- Tuning of KNN and SVM models on data extracted using PHOG:
![KNN_and_SVM_on_PHOG.png](result/KNN_and_SVM_on_PHOG.png)
- Transfer learning process of VGG16 model:
![VGG16_on_Cifar10.png](result/VGG16_on_Cifar10.png)
- Transfer learning process of YOLO11x-cls model:
![YOLO11_on_Cifar10.png](result/YOLO11_on_Cifar10.png)