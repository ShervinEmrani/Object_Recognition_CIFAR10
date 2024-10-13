
# Object Recognition on CIFAR-10

This project focuses on implementing an object recognition system using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to train a neural network model capable of accurately classifying images into one of these ten categories.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)

## Project Overview

This object recognition project utilizes the CIFAR-10 dataset to train a convolutional neural network (CNN) for image classification. The model leverages TensorFlow and Keras to build and train the network, and is designed to efficiently allocate GPU memory for training, making it adaptable to various hardware configurations.

## Dataset

The CIFAR-10 dataset contains 60,000 32x32 color images divided into 10 classes, such as airplanes, cars, birds, cats, and more. It is commonly used for benchmarking image classification models.

- **Training set**: 50,000 images
- **Test set**: 10,000 images

[here](https://www.cs.toronto.edu/~kriz/cifar.html) is the link to the CIFAR10 dataset.


## Model Architecture

The project uses a convolutional neural network (CNN) built using Keras and TensorFlow. The architecture includes:

- Convolutional layers for feature extraction
- Pooling layers to reduce spatial dimensions
- Fully connected layers for classification
- Softmax output for final predictions
- We used pre-trained ResNet50 weights for transfer learning as the backbone of our architecture. [Here](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5) is a link to the weights. 

The network is trained on the CIFAR-10 dataset and fine-tuned with hyperparameter optimization. The model also employs techniques to efficiently manage GPU memory during training.

### GPU Configuration

The model includes a configuration for GPU memory allocation:

```python
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
```

This allows the allocation of 40% of the available GPU memory to avoid memory overload.

### Notes

1. **Ensure GPU Memory**: You may need to adjust the GPU memory allocation if you are running this on a machine with limited GPU resources.
2. **Modify Architecture**: The model architecture can be easily modified to experiment with deeper networks or alternative layers to improve performance.

