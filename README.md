# CCFN:Depression-Detection-via-Multimodal-Fusion-with-Complex-valued-Capsule-Network

## Introduction

Depression is a quite common mental disorder that poses serious threats to people’s physical and mental health in the modern society. Current diagnosis strategies largely rely on doctors’ experiences and patients’ cooperation, which results in a high rate of misdiagnosis in practice. It has been noticed that people with depression exhibit typical characteristics in their expressions, speech, and other aspects, which hold significant value for diagnosing. Therefore, we propose in this paper an automatic model CCFN to aid depression detection based on multimodal characteristics of human beings. With the capsule network framework, the model uses multiple capsules to extract from single modalities fine-grained features that are then aggregated into higher-level and cross-modal capsules through a dynamic routing mechanism driven by complex numbers. Such a design allows the model to adaptively acquire semantics that can discriminate depression in the cross-modal semantic space. The experiments on a standard dataset show (1) that our model outperforms existing mainstream approaches and (2) that complex-valued capsules play a key role in the success of our model.

此代码为联合训练的多模态分类模型（语音+文本）

## 运行方式

### 1.运行特征提取代码：

    (1) cd ./CCFN/features_extract/ , 之后分别运行audio_feature_whole.py与text_feature_whole.py

### 2.运行multi_model代码: 

    (1) cd ./CCFN/multi_models/ (对应的以下模型:(CCFN,CrossCLR,Capsule,EFCNN,EFLSTM,FAL,LMF,MMLTF,TFN))

    (2) 直接测试：在对应模型(CCFN,CrossCLR,Capsule,EFCNN,EFLSTM,FAL,LMF,MMLTF,TFN)的目录下运行main.py，对应的模型保存在models里面

    (3) 三折交叉验证：运行对应模型(CCFN,CrossCLR,Capsule,EFCNN,EFLSTM,FAL,LMF,MMLTF,TFN)的目录下的main_3.py文件,在其中修改fold 1-3进行单独训练

### Please cite this if you like this code:

```
@INPROCEEDINGS{10650262,
  author={Lei, Jierui and Yang, Qingyi and Li, Bo and Zhang, Wenjian},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)}, 
  title={CCFN: Depression Detection via Multimodal Fusion with Complex-valued Capsule Network}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Adaptation models;Mental disorders;Semantics;Neural networks;Mental health;Medical services;Depression;Depression detection;Multimodal fusion;Complex capsules},
  doi={10.1109/IJCNN60899.2024.10650262}}
```
