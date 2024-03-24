# CCFN:Depression-Detection-via-Multimodal-Fusion-with-Complex-valued-Capsule-Network

此代码为联合训练的多模态分类模型（语音+文本）

## 运行方式

### 1.运行特征提取代码：

    (1) cd ./CCFN/features_extract/ , 之后分别运行audio_feature_whole.py与text_feature_whole.py

### 2.运行multi_model代码: 

    (1) cd ./CCFN/multi_models/ (对应的以下模型:(CCFN,CrossCLR,Capsule,EFCNN,EFLSTM,FAL,LMF,MMLTF,TFN))

    (2) 直接测试：在对应模型(CCFN,CrossCLR,Capsule,EFCNN,EFLSTM,FAL,LMF,MMLTF,TFN)的目录下运行main.py，对应的模型保存在models里面

    (3) 三折交叉验证：运行对应模型(CCFN,CrossCLR,Capsule,EFCNN,EFLSTM,FAL,LMF,MMLTF,TFN)的目录下的main_3.py文件,在其中修改fold 1-3进行单独训练


