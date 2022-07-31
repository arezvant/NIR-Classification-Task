# NIR Classification Task
This repository includes a project containing the PyTorch source code for the binary classification of RGB+NIR images.

The Python code structure is as follows:

    config.py
    custom_data_loader.py
    eca_mobilenetv2.py
    eca_module.py
    mobilenetv2.py
    save_checkpoint.py
    main.py
    evaluate.py
    
    
- config.py -> A default configuration which includes the global variables such as the RGB+NIR dataset root, input size of the images, and the output path to save the best trained model.
- custom_data_loader.py -> This script contains the RetiSpecDataset class which handles reading, loading, normalizing, and returning the images with their associated class id. Since the idea was to use different combinations of channels from RGB+NIR images, this class gets an input flag which handles all of the different scenarios to load required images.
- The idea for this project was to use various approaches in order to utilize the NIR channel efficiently. MobileNetV2 architecture has been used for all of the approaches as the base architecture which can be found in mobilenetv2.py file.
  - The first approach was to use Efficient Channel Attention for Deep Convolutional Neural Networks (ECANet) [1] (4 channels).
    - RetiSpecClass dataset handles the preparation of this version of dataset with the input flag "att-chan".
    - The implementation of ECA module can be found in eca_module.py file. 
    - The implementaion for the integration of ECA module with MobileNetV2 architecture can be found in eca_mobilenetv2.py file as well.
  - Another approach was to randomly drop one of the RGB channels and use the rest with the NIR channel (3 channels). 
    - RetiSpecClass dataset handles the preparation of this version of dataset with the input flag "mixed-chan".
    - Since there are 3 channels, MobileNetV2 has been used as the main architecture for this approach.
  - As for the base approach, RGB channel is separated from the NIR channel, and each one of these channels are trained separately on the MobileNetV2 architecture to see the performance of each individual channel as for the baseline comparison.
- save_checkpoint.py -> This script contains the code to save the best weight and model for each approach in order to use them later for the evaluation purposes.
- main.py -> This script handles all of the training purposes. The main () function has different parameteres which you can set. These inlcude:
  - training root, validation root
  - batch size (bs)
  - an approach to train the model (arch)
  - learning rate (lr)
  - momentum
  - weight decay
  - epochs 
- The main starts with initializing the data augmentation pipeline. Since the raw performance was good enough, this piepline was never used during our experiments. Then, we have the initialization of the dataset and data loaders based on the approache that we have specified as the arch parameter to the main ().
- In each training approach, the MLflow dashboard has been used to monitor training/validation accuracy/loss values.- 
- evaluate.py -> This function has been called in the training phase in each approach. It handles loading process of the best trained model and making a prediction on the given validation/test images. This function outputs the major binary classification metrics such as total average accuracy, recall, specificty, precision, f1-score, and Confusion Matrix for the selected model under the selected approach. Then, all of these metrics with the arch paramerter will be logged onto the MLflow dashboard.

For running purrposes, add a Data folder with RGB+NIR images inside. Then, run the main.py with the required parameteres.
A test result from the MLflow dashboard can be seeb as follows:
![Result](https://github.com/arezvant/NIR-Classification-Task/blob/main/Test%20Results.PNG)

# Citation
<a id="1">[1]</a> 
Wang, Q., et al. "ECA-Net: Efficient channel attention for deep convolutional neural networks. In 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR)(pp. 11531–11539)." IEEE. https://ieeexplore.ieee.org/document/9156697 (2020).
