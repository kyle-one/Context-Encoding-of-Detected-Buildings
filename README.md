# Context-Encoding-of-Detected-Buildings
Bounding Boxes Are All We Need: Street View Image Classification via Context Encoding of Detected Buildings

[arxiv](https://arxiv.org/abs/2010.01305)
# Introduction 
1. In this paper, an approach based on a detector-encoder-classifier framework is proposed. Different from common end-to-end models, our approach does not use visual features of the whole image directly. The proposed framework obtains the bounding boxes of buildings in street view images from a detector. Their contextual information such as building classes and positions are then encoded into metadata and finally classified by a recurrent neural network. 
2. To verify our approach, we made a dataset of 19,070 street view images and 38,857 buildings based on the BIC_GSV dataset through a combination of automatic label acquisition and expert annotation. The dataset can be used not only for street view image classification aiming at urban land use analysis, but also for multi-class building detection. Experiments show that the proposed method achieves significant performance improvement over the models based on end-to-end convolutional neural network. 
# Results
|Models|M-P|M-R|M-F1|
| :--:|:--:|:--:|:--:|
|layout+perfect detector	|95.54	|92.15|	93.82|
|layout+Ca101 best	|81.81|	80.94	|81.37|
| co-occurrence+Ca101 best | 81.47	|80.53|	81.00|
|baseline:ResNet50|	69.16	|68.94	|69.05|
* Bulidings bounding bboxs were genarated using [MMDetection](https://github.com/open-mmlab/mmdetection/) with their default hyperparameters and pre-trained models.
# BEAUTY Dataset 
[Download BEAUTY Dataset](https://drive.google.com/file/d/15gHUUwbPVD_JEgdYCSWzKMxjOlmRS5qC/view?usp=sharing)
  
**BEAUTY(Building dEtection And Urban funcTional-zone portraYing):** A street view image dataset with a dual label system is made based on the exist BIC_GSV dataset.

  
The authors would like to thank the authors of reference [1] for publishing the BIC GSV dataset including city scale GSV images. We would also like to thank Mengshuo Fan and Zhiwei He, the experts in architecture and urban planning from the BIM Research Center, Qingdao Research Institute of Urban and Rural Construction for their professional guidance on manual annotation. Thanks to those who participated in manual annotation for building detection: Yu Ma, Shanshan Lin, Ying Guo and Kaixin Li, and who participated in manual annotation for street view image classification: Ying Zhang, Jiaojie Wang, Shujing Ma and Yue Wang.
  

 
 
[1] J. Kang, M. Korner, Y. Wang, H. Taubenb ¨ ock, and X. X. Zhu, “Building ¨ instance classification using street view images,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 145, pp. 44–59, 2018.
