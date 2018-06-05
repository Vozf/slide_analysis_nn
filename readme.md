# Slide Analysis Neural Network
## Slide Analysis Neural Network is part of the [Slide Analysis Project](https://github.com/Vozf/slide_analysis_web)
### Data (200+ Gb)
Input data are histology whole slide images(very large images with size 2-3Gb size) and [ASAP](https://github.com/computationalpathologygroup/ASAP) annotations. Annotations contain markup of the tumor area in the image. [Example 1](https://psv4.userapi.com/c848232/u98389977/docs/d12/01d5273dbd9e/Screenshot_from_2018-06-05_12-15-50.png?extra=Ak5n4Vmmtz3ntEI1E8HyLOe9whCUltmMvm6jzVTF_-n5xIKESXZgRc3kTt6Un7bEnhm413k30_VPSp5s5n7NdgfK-R-PnBTEKqaVvcTWGV0X0ihOaTqTgkj_GF7peeiTi-k1dIPI8AUyoeg) [Example 2](https://psv4.userapi.com/c848232/u98389977/docs/d6/767f44e4726e/Screenshot_from_2018-06-05_12-19-19.png?extra=XdMoLxiLriBuyBwpBBiwo0Tw9jOf9dEmfFZML8xqFCvL37lc_TjLv-xT70QJ_WRpk2B9TUUhKy_peAs6NOlTRgmMAMRpnS-YQAU4aGMnpO5UIJizesIWf1LsTgXPI10blLPJ101m5F-U2Wg) (Blue markup is the tumor area marked by the medical specialist)

### Target 
Task of the neural network is image segmentation on the whole slide images. The task is reduced to image classification due to the size of the images which makes it available to look through the subimages. Service can output results in either raw type or it can produce it on top of [ASAP](https://github.com/computationalpathologygroup/ASAP) annotations 
### Tools
InceptionV3 is used as neural network and Keras is used as framework. 
### Examples of prediction on top of ASAP annotations
Red squares are the tumor area predicted by the neural network and as mentioned above blue markup is the tumor area marked by the medical specialist.
[Example 1](https://psv4.userapi.com/c848232/u98389977/docs/d6/ee75c2427cbb/Screenshot_from_2018-06-05_11-53-12.png?extra=c0mcb9Cnd8I0iypXGuH6BRIh1zI3cMg6M7_eBrZo5fjhF0bzYLnNqt9vjjHCzjr8Y4teWTIpnFImg12fHZlDrmRwMM7dRearKcEMPeF45iUU_9WfjbsDdVUwSG89FBetGOvDmS7CG5tZcz4)
[Example 2](https://psv4.userapi.com/c848232/u98389977/docs/d15/221cec219120/Screenshot_from_2018-06-05_11-54-55.png?extra=2BcnoH2mro4UWyX-5z0T375UHVuks37ZNDhiKNqBFYEnnplFC_miGGczzZ8i3mrVDyBwd31_hK1MbZWG_UQCsrjTtVRfA_N5PYptRMbNWO0g_hbbVmqA7fuADd2lNh0ApC7MGcaHg6QJZPk)
[Example 3](https://psv4.userapi.com/c848232/u98389977/docs/d14/35438ab1ee8d/Screenshot_from_2018-06-05_12-25-20.png?extra=5g6B62s-gX9_F8drbC-rjY90-2ZHmwVkJ6o3ZQujKwz7RXHWozj6n_gYjyexoNXMLXO5caJruAHN6ZEXfQA1Lxd-YMBve20Rp__YqCoYOI-8jD2fkDmOoNtn4pPJYPPR6IoLLIK_bfGPdB4)
### Use
You can use the neural network as part of the [Slide Analysis Project](https://github.com/Vozf/slide_analysis_web) or you can use the repository itself and use `Predict` class for prediction. Weights are auto downloaded.
Training and Prediction are also available using [Nvidia docker](https://github.com/NVIDIA/nvidia-docker)