
# In[]
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch 
from transformers import BertTokenizer
from transformers import BertModel
# In[]
topics = ['positive', 'neutral', 'negative']
answers = {}
text_features = []
text_targets = []
ss=[]
# In[]
import numpy as np
import os
from tqdm import tqdm
tokenizer = BertTokenizer.from_pretrained(os.path.dirname(os.getcwd())+'/bert-base-chinese')
model = BertModel.from_pretrained(os.path.dirname(os.getcwd())+'/bert-base-chinese',output_hidden_states = True)#正式导入中文bert
def extract_features(text_features, text_targets, path,tokenizer=tokenizer,model=model):
    for index in tqdm(range(114)):
        if os.path.isdir(os.path.dirname(os.getcwd())+"/EATD-Corpus/"+str(path)+str(index+1)):
            answers[index+1] = []
            for topic in topics:
                with open(os.path.dirname(os.getcwd())+"/EATD-Corpus/"+str(path)+str(index+1)+"/"+'%s.txt'%(topic) ,'r',encoding="utf-8") as f:
                    lines = f.readlines()[0]
                    marked_text = "[CLS] " + lines + " [SEP]"	#开头，结尾	
                    print(marked_text)
                    #tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')#导入中文bert	
                    tokenized_text = tokenizer.tokenize(marked_text) #对句子进行分词	
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)	#得到每个词在词表中的索引
                    segments_ids = [1] * len(tokenized_text)	#分词，精确到个体
                    tokens_tensor = torch.tensor([indexed_tokens])#得到每个词在词表中的索引的tensor
                    
                    segments_tensors = torch.tensor([segments_ids])#分词tensor	
                    #print(segments_tensors.shape)
                    #model = BertModel.from_pretrained('bert-base-chinese',
                    #                                  output_hidden_states = True)#正式导入中文bert
                    model.eval()
                    with torch.no_grad():
                        outputs = model(tokens_tensor, segments_tensors)
                        hidden_states = outputs[2]
                    #print(hidden_states[11].shape)
                    
                    token_embeddings = torch.stack(hidden_states, dim=0)
                    #print(token_embeddings.size())
                    token_embeddings = torch.squeeze(token_embeddings, dim=1)
                    #print(token_embeddings.size())
                    token_embeddings = token_embeddings.permute(1,0,2)#调换顺序
                    #print(token_embeddings.size())
                    
                    #句子向量表示
                    token_vecs = hidden_states[-2][0]
                    sentence_embedding = torch.mean(token_vecs, dim=0)#一个句子就是768维度
                    answers[index+1].append(sentence_embedding)
            with open(os.path.dirname(os.getcwd())+'/EATD-Corpus/'+str(path)+str(index+1)+'/new_label.txt'.format(index+1, path)) as fli:
                target = float(fli.readline())
            text_targets.append(1 if target >= 53 else 0)
            temp=[]
            for i in range(3):
                temp.append(np.array(answers[index+1][i]))
            text_features.append(temp)
# In[]
extract_features(text_features, text_targets, 't_')
# In[]
extract_features(text_features, text_targets, 'v_')
# In[]
print("Saving npz file locally...")
text_features = np.array(text_features)
text_targets = np.array(text_targets)
print(text_features.shape,text_targets.shape)
np.savez(os.path.dirname(os.getcwd())+'/text_feature.npz', text_features)
np.savez(os.path.dirname(os.getcwd())+'/text_target.npz', text_targets)