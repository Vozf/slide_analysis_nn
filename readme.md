# Slide Analysis Neural Network
## Slide Analysis Neural Network is part of the [Slide Analysis Project](https://github.com/Vozf/slide_analysis_web)
### Data (200+ Gb)
Input data are histology whole slide images(very large images with size 2-3Gb size) and [ASAP](https://github.com/computationalpathologygroup/ASAP) annotations. Annotations contain markup of the tumor area in the image. [Example 1](https://vk.com/doc98389977_467637803) [Example 2](https://vk.com/doc98389977_467638031) (Blue markup is the tumor area marked by the medical specialist)

### Target 
Task of the neural network is image segmentation on the whole slide images. The task is reduced to image classification due to the size of the images which makes it available to look through the subimages. Service can output results in either raw type or it can produce it on top of [ASAP](https://github.com/computationalpathologygroup/ASAP) annotations 
### Tools
InceptionV3 is used as neural network and Keras is used as framework. 
### Examples of prediction on top of ASAP annotations
Red squares are the tumor area predicted by the neural network and as mentioned above blue markup is the tumor area marked by the medical specialist.
[Example 1](https://vk.com/doc98389977_467638148)
[Example 2](https://vk.com/doc98389977_467638149)
[Example 3](https://vk.com/doc98389977_467638496)
### Use
You can use the neural network as part of the [Slide Analysis Project](https://github.com/Vozf/slide_analysis_web) or you can use the repository itself and use `Predict` class for prediction. Weights are auto downloaded.
Training and Prediction are also available using [Nvidia docker](https://github.com/NVIDIA/nvidia-docker)
