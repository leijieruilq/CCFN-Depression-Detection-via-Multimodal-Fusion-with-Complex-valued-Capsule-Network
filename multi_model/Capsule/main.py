import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from Capsule_fusion import CapsuleNet
from Capsule import CapsuleLoss
import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import itertools
# In[]

def load_audio_features():
    audio_features = np.squeeze(np.load(os.path.dirname(os.path.dirname(os.getcwd()))+"/audio_feature.npz")['arr_0'], axis=2)
    audio_targets = np.load(os.path.dirname(os.path.dirname(os.getcwd()))+"/audio_target.npz")['arr_0']
    audio_targets[audio_targets>=53]=1
    audio_targets[(audio_targets<53)&(audio_targets!=1)]=0
    return audio_features,audio_targets

def load_text_features():
    text_features = np.load(os.path.dirname(os.path.dirname(os.getcwd()))+"/text_feature.npz")['arr_0']
    text_targets = np.load(os.path.dirname(os.path.dirname(os.getcwd()))+"/text_target.npz")['arr_0']
    return text_features,text_targets

def train_test_split_done():
    audio_x,y = load_audio_features()
    text_x,_ = load_text_features()
    #扩充数据集
    for i in range(y.shape[0]):
        if y[i]==1:
            feat_audio = audio_x[i]
            feat_text = text_x[i]
            count = 0
            resample_idxs = [1,2,3,4,5]
            for (i,j) in zip(itertools.permutations(feat_text,feat_text.shape[0]),
                             itertools.permutations(feat_audio,feat_audio.shape[0])):
                count = count + 1
                if count in resample_idxs:
                    audio_x = np.vstack((audio_x, np.expand_dims(list(j), 0)))
                    text_x = np.vstack((text_x, np.expand_dims(list(i), 0)))
                    y = np.hstack((y,1))
        else:
            feat_audio = audio_x[i]
            feat_text = text_x[i]
            count = 0
            resample_idxs = [5]
            for (i,j) in zip(itertools.permutations(feat_text,feat_text.shape[0]),
                             itertools.permutations(feat_audio,feat_audio.shape[0])):
                count = count + 1
                if count in resample_idxs:
                    audio_x = np.vstack((audio_x, np.expand_dims(list(j), 0)))
                    text_x = np.vstack((text_x, np.expand_dims(list(i), 0)))
                    y = np.hstack((y,0))
    # 使用 train_test_split 分别划分 x1 和 x2，同时确保 y 的划分与它们保持一致  
    x1_train, x1_test, y_train, y_test = train_test_split(audio_x, y, test_size=0.2, random_state=42,stratify=y)  
    x2_train, x2_test, _, _ = train_test_split(text_x, y, test_size=0.2, random_state=42,stratify=y)    
    return x1_train, x2_train, y_train,\
           x1_test, x2_test, y_test

def train_and_test():
    x1_train, x2_train, y_train,x1_test, x2_test, y_test=train_test_split_done()

def get_param_group(model):
    nd_list = []
    param_list = []
    for name, param in model.named_parameters():
        if 'ln' in name:
            nd_list.append(param)
        else:
            param_list.append(param)
    return [{'params': param_list, 'weight_decay': 1e-5}, {'params': nd_list, 'weight_decay': 0}]

def train(epoch,model,config,xtrain1,xtrain2,ytrain,criterion,device):
    global lr, train_acc
    model.train()
    batch_idx = 1
    total_loss = 0
    correct = 0
    for i in tqdm(range(0, xtrain1.shape[0],config['batch_size'])):
        if i + config['batch_size'] > xtrain1.shape[0]:
            x1, x2, y = xtrain1[i:],\
                        xtrain2[i:],\
                        ytrain[i:]
        else:
            x1, x2, y = xtrain1[i:(i + config['batch_size'])],\
                        xtrain2[i:(i + config['batch_size'])],\
                        ytrain[i:(i + config['batch_size'])]

        if config['cuda']:
            x1,x2,y = Variable(torch.from_numpy(x1).type(torch.FloatTensor), requires_grad=True).to(config['device']),\
                      Variable(torch.from_numpy(x2).type(torch.FloatTensor), requires_grad=True).to(config['device']),\
                      Variable(torch.from_numpy(y)).to(config['device'])
        else:
            x1,x2,y = Variable(torch.from_numpy(x1).type(torch.FloatTensor), requires_grad=True),\
                         Variable(torch.from_numpy(x2).type(torch.FloatTensor), requires_grad=True),\
                         Variable(torch.from_numpy(y))

        # 将模型的参数梯度设置为0
        optimizer.zero_grad()
        output,_ = model(x1,x2)
        pred = output.data.max(1, keepdim=True)[1]
        #print(pred.shape, y.shape)
        correct += pred.eq(y.data.view_as(pred)).cuda().sum()
        loss = criterion(output,y.long())
        # 后向传播调整参数
        loss.backward()
        total_loss += loss.item()
        # 根据梯度更新网络参数
        optimizer.step()
        # loss.item()能够得到张量中的元素值
    train_acc = correct
    print(
        'Train Epoch: {:2d}\t Learning rate: {:.4f}\tLoss: {:.6f}\t Accuracy: {}/{} ({:.0f}%)\n '
        .format(epoch + 1, config['learning_rate'], total_loss/config["batch_size"], correct,
                xtrain1.shape[0], 100. * correct / xtrain1.shape[0]))
    return total_loss
# In[]
def standard_confusion_matrix(y_test, y_test_pred):
    return confusion_matrix(y_test.cpu().numpy(), y_test_pred)


def model_performance(y_test, y_test_pred_proba):
    y_test_pred = y_test_pred_proba.data.max(1, keepdim=True)[1]
    conf_matrix = standard_confusion_matrix(y_test, y_test_pred.cpu().numpy())
    return y_test_pred, conf_matrix
# In[]
def evaluate(model,config,xtest1,xtest2,ytest):
    model.eval()
    accuracy=0
    precision=0
    recall=0
    f1_score=0
    cmax=0
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(0, xtest1.shape[0],config['batch_size'])):
            if i + config['batch_size'] > xtest1.shape[0]:
                x1, x2, y = xtest1[i:],xtest2[i:],ytest[i:]
            else:
                x1, x2, y = xtest1[i:(i + config['batch_size'])],\
                            xtest2[i:(i + config['batch_size'])],\
                            ytest[i:(i + config['batch_size'])]
            if config['cuda']:
                x1,x2,y = Variable(torch.from_numpy(x1).type(torch.FloatTensor), requires_grad=True).to(config['device']),\
                          Variable(torch.from_numpy(x2).type(torch.FloatTensor), requires_grad=True).to(config['device']),\
                          Variable(torch.from_numpy(y)).to(config['device'])
            else:
                x1,x2,y = Variable(torch.from_numpy(x1).type(torch.FloatTensor), requires_grad=True),\
                             Variable(torch.from_numpy(x2).type(torch.FloatTensor), requires_grad=True),\
                             Variable(torch.from_numpy(y))
            optimizer.zero_grad()
            output,_ = model(x1,x2)
            y_test_pred, conf_matrix = model_performance(y, output.cuda())
            cmax+=conf_matrix
        print("Confusion Matrix:")
        print(cmax)
        accuracy = float(cmax[0][0] + cmax[1][1]) / np.sum(cmax)
        precision = float(cmax[0][0]) / (cmax[0][0] + cmax[0][1])
        recall = float(cmax[0][0]) / (cmax[0][0] + cmax[1][0])
        f1_score = 2 * (precision * recall) / (precision + recall)
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1-Score: {}\n".format(f1_score))
        print('=' * 89)
    return accuracy,precision,recall,f1_score
# In[]
if __name__ == '__main__':
    configs_audio = {"audio_c_in":3,
                     "audio_c_out":320}
    configs_text = {'text_dropout':0.3,
                    'text_rnn_layers': 2,
                    'text_hidden_dims':320,
                    'text_embedding_size': 768,
                    'text_bidirectional': True}
    configs_fusion = {"fusion_in":2,
                      "fusion_c_in":8,
                      "num_conv_units":32,
                      "fusion_c_out":8,
                      "fusion_out_dim":16,
                      "num_classes":2,
                      "fusion_dim":640,
                      "device":"cuda:4"}
    config_exp = {"batch_size":32,
                  "cuda":True,
                  "device":"cuda:4",
                  'learning_rate':1e-3,
                  "patience":200, #早停轮数
                  "epochs":250}
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = train_test_split_done()
    print(y_train.shape,y_test.shape)
    model = CapsuleNet(configs_audio,configs_text,configs_fusion).to(config_exp["device"])
    optimizer = optim.Adam(model.parameters(), lr=config_exp["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    loss_text=[]
    acc=[]
    pre=[]
    recalls=[]
    f1=[]
    best_f1_score=0
    best_acc_score=0
    early_stop_counter=0
    for ep in range(1, config_exp['epochs']):
        tloss = train(ep,model,config_exp,xtrain1=x1_train,xtrain2=x2_train,ytrain=y_train,criterion=criterion,
                      device=config_exp["device"])
        accuracy,precision,recall,f1_score=evaluate(model,config_exp,xtest1=x1_test,xtest2=x2_test,ytest=y_test)
        loss_text.append(tloss)
        acc.append(accuracy)
        pre.append(precision)
        recalls.append(recall)
        f1.append(f1_score)
        if f1_score>best_f1_score and accuracy>best_acc_score:
            best_f1_score=f1_score
            best_acc_score=accuracy
            print("current best:(acc,precision,recall,f1)",accuracy,precision,recall,f1_score)
        else:  
            early_stop_counter += 1  
            if early_stop_counter >= config_exp["patience"]:  
                print("Early stopping at epoch", ep)  
                break
        if best_f1_score>=0.85:
            torch.save(model,os.path.dirname(os.path.dirname(os.getcwd()))+"/models/Capsule_models/fuse-"+str(np.round(best_f1_score,2)))