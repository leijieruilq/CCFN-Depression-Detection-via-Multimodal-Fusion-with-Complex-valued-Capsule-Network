# CCFN:Depression-Detection-via-Multimodal-Fusion-with-Complex-valued-Capsule-Network

## Introduction

Depression is a quite common mental disorder that poses serious threats to people’s physical and mental health in the modern society. Current diagnosis strategies largely rely on doctors’ experiences and patients’ cooperation, which results in a high rate of misdiagnosis in practice. It has been noticed that people with depression exhibit typical characteristics in their expressions, speech, and other aspects, which hold significant value for diagnosing. Therefore, we propose in this paper an automatic model CCFN to aid depression detection based on multimodal characteristics of human beings. With the capsule network framework, the model uses multiple capsules to extract from single modalities fine-grained features that are then aggregated into higher-level and cross-modal capsules through a dynamic routing mechanism driven by complex numbers. Such a design allows the model to adaptively acquire semantics that can discriminate depression in the cross-modal semantic space. The experiments on a standard dataset show (1) that our model outperforms existing mainstream approaches and (2) that complex-valued capsules play a key role in the success of our model.

This code implements a jointly trained multimodal classification model (audio + text).

## Execution Procedure

### 1. Run feature extraction scripts:

    (1) cd ./CCFN/features_extract/ , then execute audio_feature_whole.py and text_feature_whole.py sequentially

### 2. Run the multi_model code:

    (1) cd ./CCFN/multi_models/ (corresponding to the following models: (CCFN, CrossCLR, Capsule, EFCNN, EFLSTM, FAL, LMF, MMLTF, TFN))

    (2) Direct testing: Run main.py in the directory of the corresponding model (CCFN, CrossCLR, Capsule, EFCNN, EFLSTM, FAL, LMF, MMLTF, TFN). The models are stored in the models directory.

    (3) Three-fold cross-validation: Run the main_3.py file in the directory of the corresponding model (CCFN, CrossCLR, Capsule, EFCNN, EFLSTM, FAL, LMF, MMLTF, TFN). Modify fold 1-3 within the file to train each fold separately.

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
