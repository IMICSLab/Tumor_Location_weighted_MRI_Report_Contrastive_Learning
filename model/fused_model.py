import numpy as np
import pandas as pd
import torch
import random
import socket
import torch.nn as nn
import time
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from functools import partial
import torch.nn.functional as F
import sys
from sklearn.model_selection import train_test_split
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import copy
from transformers import AutoModel, AutoTokenizer
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig,LongformerModel,LongformerForMaskedLM
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import metrics
from torch.autograd import Variable
# transformers
class BertClassifier(nn.Module): #(GPT2ForSequenceClassification):#

    def __init__(self, freeze_bert = True,dropout=0.25):

        super(BertClassifier, self).__init__()

        #self.config = AutoConfig.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
        # self.bert_layer =AutoModel.from_pretrained("ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section")#GPT2Model.from_pretrained("healx/gpt-2-pubmed-medium")##AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-BigBird")# #AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m', config=self.config) #BertModel.from_pretrained('bert-base-uncased')#AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m', config=self.config)#
        ######model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="microsoft/biogpt")#"healx/gpt-2-pubmed-medium", num_labels=2)
#gpt2
        # self.bert_layer=GPT2ForSequenceClassification.from_pretrained("microsoft/biogpt")#('gpt2')#("healx/gpt-2-pubmed-medium",num_labels=2)#, id2label=id2label, label2id=label2id)#GPT2Model
        # self.bert_layer=AutoModel.from_pretrained("allenai/longformer-base-4096")#TransformerLanguageModel.from_pretrained(
        self.bert_layer = LongformerForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer",#'allenai/longformer-base-4096',
                                                        gradient_checkpointing=False,
                                                        attention_window = 512)
                                                         
       
        # fix model padding token id
        # self.bert_layer.config.pad_token_id = self.bert_layer.config.eos_token_id#############################333
        config = LongformerConfig()
        
        # self.bert_layer.score=nn.Linear(768, 1)#################
        self.bert_layer.classifier.out_proj=nn.Linear(768, 256)
        #Freeze bert layers
        for param in self.bert_layer.parameters():
            
            param.requires_grad = False
        for param in self.bert_layer.longformer.encoder.layer[11].parameters():#score #pooler
            param.requires_grad = True
        # for param in self.bert_layer.classifier.parameters():#score #pooler
        #     param.requires_grad = True
        for param in self.bert_layer.longformer.encoder.layer[10].parameters():#score #pooler
            param.requires_grad = True
        # for param in self.bert_layer.transformer.h[23].parameters():
        #     param.requires_grad = True
        # for param in self.bert_layer.transformer.h[22].parameters():
        #     param.requires_grad = True
        # for param in self.bert_layer.transformer.h[11].parameters():
        #     param.requires_grad = True
        # for param in self.bert_layer.transformer.h[10].parameters():
        #     param.requires_grad = True
        # self.bert_layer2=deleteEncodingLayers(self.bert_layer)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(0.3)
        # self.linear = nn.Linear(768, 1)
        # self.test=nn.Linear(1024, 768)


        # self.fc1=nn.Linear(1024, 512)
        # self.fc3=nn.Linear(512,256)
        # # self.fc1=nn.Linear(524288, 1024)
        # # self.fc4=nn.Linear(256,128)
        # self.fc2=nn.Linear(256, 1)


        # self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask = None):

        # print(self.bert_layer.transformer.layer[-1])
        # print(self.bert_layer.transformer.layer[:-1])
        # pooled_output,_= self.bert_layer(input_ids= input_ids, attention_mask=attention_mask,return_dict=False)



        # attention_mask = (input_ids != 0).float() 

        pooled_output=self.bert_layer(input_ids= input_ids, attention_mask=attention_mask,return_dict=True).logits#######
        # _,pooled_output=self.bert_layer(input_ids= input_ids, attention_mask=attention_mask,return_dict=False)[:2]
        # print(pooled_output.shape)
        # pooled_output=pooled_output[0][..., :-2]
        # print(pooled_output.shape)
        # pooled_output=nn.Sequential(*list(pooled_output.children())[:-1])
        # print(pooled_output)
        # print(self.bert_layer(input_ids= input_id, attention_mask=mask,return_dict=False))
        #
        # pooled_output,_= self.bert_layer(input_ids= input_id, attention_mask=mask,return_dict=False)
        
        dropout_output = self.dropout1(pooled_output)
        # dropout_output = self.fc1(dropout_output)#.view(dropout_output.shape[0],-1))
        # dropout_output = pooled_output
        # dropout_output = self.fc2(dropout_output)
        #####
        # linear_output = self.linear(dropout_output)
        # print(dropout_output.shape)
        # dropout_output = self.fc2(dropout_output)
        # final_layer = self.relu(linear_output)
        # print(final_layer.shape)
        # print(pooled_output[1].shape)
        return dropout_output#dropout_output








# RESNET  #from here


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.name = "BasicBlock"
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.name = "Bottleneck"
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 model_depth,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.name = f"ResNet_pLGG_Classifer_depth{model_depth}"

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.fc = nn.Linear(512, 1)
        
        dropout_rate=0.25#0.1 #0.15
        self.dropout = nn.Dropout(dropout_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        # self.in_planes = planes * block.expansion  
        layers.append(
             block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion  #was here
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.dropout(x)
        
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        local=x
        x = self.layer4(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        
        # x = self.fc(x)
 #       x = torch.sigmoid(x)#softmax(x)

        return x,local


def generate_model(model_depth, inplanes, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    model = None
    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], inplanes, model_depth, **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], inplanes, model_depth, widen_factor = 0.5, **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], inplanes, model_depth, **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], inplanes, model_depth, **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], inplanes, model_depth, **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], inplanes, model_depth, **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], inplanes, model_depth, **kwargs)

    return model
# to here

######################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444
# def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
#     # 3x3x3 convolution with padding
#     return nn.Conv3d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         dilation=dilation,
#         stride=stride,
#         padding=dilation,
#         bias=False)


# def downsample_basic_block(x, planes, stride, no_cuda=False):
#     out = F.avg_pool3d(x, kernel_size=1, stride=stride)
#     zero_pads = torch.Tensor(
#         out.size(0), planes - out.size(1), out.size(2), out.size(3),
#         out.size(4)).zero_()
#     if not no_cuda:
#         if isinstance(out.data, torch.cuda.FloatTensor):
#             zero_pads = zero_pads.cuda()

#     out = Variable(torch.cat([out.data, zero_pads], dim=1))

#     return out


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.dilation = dilation

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = nn.Conv3d(
#             planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm3d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.dilation = dilation

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

###$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$##################################33
class ResNet_attention(nn.Module):

    def __init__(self,use_attention,
                 block,
                 layers,
                 sample_input_D=14,
                 sample_input_H=28,
                 sample_input_W=28,
                 shortcut_type='A',
                 mode="pretraining"):
        self.inplanes = 64
        
        super(ResNet_attention, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.mode = mode
        self.use_attention = use_attention
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        
        self.text_feat_dim: int = 768
        # self.output_dim: int = 768,
        self.output_dim=512
        self.hidden_dim: int = 256#2048
        self.interm_feature_dim=256#1024
        # self.depth_attention = DepthAttention(512)
        self.feature_dim = 512
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.local_embed = LocalEmbedding_3d(
            self.interm_feature_dim, self.hidden_dim, self.output_dim
        )


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.mode=="pretraining":
            return self.forward_pretraining(x)
        elif self.mode=="downstream":
            return self.forward_downstream(x)

    def forward_pretraining(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        local_featrues = x
        x = self.layer4(x)
        x , _= self.self_attention(x)
        
        x = self.avgpool (x)
        
        x = x.view(x.size(0), -1)  #####changed for 3d
        
        return x,local_features.contiguous()

        
    
    def forward_downstream(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.use_attention==True:
            #newly added
            # x = x.mean(1)


            #newly added end
            # x  =  self.depth_attention(x)
            
            x ,atten_weights= self.self_attention(x)#(x,x,x,need_weights=True)  #, atten_weights
            # print("whyyyy",atten_weights.shape,atten_weights2.shape)
            # print("avg_pool",x.shape)
            # x=self.avgpool (x)
            
            x = x.view(x.size(0), -1) 
            
            return x,atten_weights
        else:
            x=self.avgpool (x)
            # print("xxxxx",x.shape)
            x=x.view(x.size(0),-1)
            # x = torch.flatten(x, 1)
            return x

        return x




class image_text(torch.nn.Module):
    def __init__(self):
        super().__init__()

        inplanes = [ 128, 256, 512,1024]#[64, 128, 256, 512]
        inplanes = [64, 128, 256, 512]
        # self.cnn=generate_model(model_depth=18, n_classes=1039,inplanes=inplanes)	
        self.cnn = ResNet_attention2(True,BasicBlock,[2,2,2,2],model_depth=18, n_classes=1039,block_inplanes=inplanes,output_dim=256,mode="downstream")
        
        self.cnn.conv1 = nn.Conv3d(1, 64,kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
		# net.fc = net.fc = nn.Linear(512, 1)#3) #512		
        # Below lines are for sentence embedding which has size 150 for self_trained_embedding
       
        self.transformer=BertClassifier()
        
        # Adding intermediate linear layer before final classification
        num_intermediate_output = 256
        self.intermediate_layer = nn.Linear(256 + 256 + 256, num_intermediate_output)

        # Adding a final classification layer that takes in outputs 3
        self.final_classification_layer = nn.Linear(num_intermediate_output, 1)

		
    def forward(self, image, input_ids, attention_mask = None):
				                    
        img_emb,attn = self.cnn(image)
                
        # Pass sentence embedding through linear layers
        text_emb = self.transformer(input_ids, attention_mask)
        # print("final_size",img_emb.shape,text_emb.shape)
        combined = torch.cat((img_emb, text_emb), axis=1)
        intermediate = self.intermediate_layer(combined)
        
        # Get final labels
        labels = self.final_classification_layer(intermediate)

        
        return  labels,attn




###############################################################################################################


# class GlobalEmbedding(nn.Module):
#     def __init__(self,
#                  input_dim,#: int = 768,
#                  hidden_dim,#: int = 2048,
#                  output_dim):#: int = 512) -> None:
#         super().__init__()
#         # print("input_dim",input_dim)
#         # print("hidden_dim",hidden_dim)
#         # print("output_dim",output_dim)
#         # self.head = nn.Sequential(
#         #     # nn.Linear(input_dim, hidden_dim),
#         #     # nn.BatchNorm1d(hidden_dim),
#         #     # nn.ReLU(inplace=True),
#         #     nn.Linear(input_dim, output_dim),
#         #     nn.BatchNorm1d(output_dim, affine=False)  # output layer
#         # )
#         # 2 linears
#         # self.head = nn.Sequential(  
#         #     nn.Linear(input_dim, 256),
#         #     nn.BatchNorm1d(256),
#         #     nn.ReLU(inplace=True),
#         #     nn.Linear(256, output_dim),
#         #     nn.BatchNorm1d(output_dim, affine=False)  # output layer
#         # end 2 linears
        
#         self.head = nn.Sequential(  
#             nn.Linear(input_dim, 256),#,#256
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128, affine=False),  # output layer
#             nn.ReLU(inplace=True) , #remove
#             nn.Linear(128, 64)
#         )

#     def forward(self, x):
#         # print("xxxx",x.shape)
#         # print("input_dim",input_dim)
#         # print("hidden_dim",hidden_dim)
#         # print("output_dim",output_dim)
#         return self.head (x )

class SoftAttention(nn.Module):
    def __init__(self,in_groups,m_heads,in_channels):
        super(SoftAttention, self).__init__()
        
        self.learnable_scalar = nn.Parameter(torch.rand(1))
        self.conv3d = nn.Conv3d(in_channels=in_groups,out_channels=m_heads,kernel_size=(in_channels,1,1), stride=(in_channels,1,1))
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
#         print('x.shape:',x)
        h,w = x.shape[-2],x.shape[-1]
        c = torch.unsqueeze(x,1)
#         print('c.shape:',c)
        c = self.conv3d(c)
        c = self.lrelu(c)
#         print('c.shape relu:',c)
        c = c.squeeze(2)
        c = c.view(c.shape[0],c.shape[1],h*w)
#         print('c.shape h*w:',c)
        c = self.softmax(c)
#         print('c.shape sfmx:',c)
        c = c.view(c.shape[0],c.shape[1],h,w)
#         print('c.shape:',c)
        attn_maps = torch.unsqueeze(c.sum(1),1)
#         print('attn_maps.shape:',attn_maps)
        importance = x*attn_maps
        out = x + importance*self.learnable_scalar.expand_as(importance)
#         print('out.shape:',out)
        return out, attn_maps, self.learnable_scalar

# class GlobalEmbedding(nn.Module):
#     def __init__(self,
#                  input_dim,#: int = 768,
#                  hidden_dim,#: int = 2048,
#                  output_dim):#: int = 512) -> None:
#         super().__init__()
#         # print("input_dim",input_dim)
#         # print("hidden_dim",hidden_dim)
#         # print("output_dim",output_dim)
#         # self.head = nn.Sequential(
#         #     # nn.Linear(input_dim, hidden_dim),
#         #     # nn.BatchNorm1d(hidden_dim),
#         #     # nn.ReLU(inplace=True),
#         #     nn.Linear(input_dim, output_dim),
#         #     nn.BatchNorm1d(output_dim, affine=False)  # output layer
#         # )
#         # 2 linears
#         # self.head = nn.Sequential(  
#         #     nn.Linear(input_dim, 256),
#         #     nn.BatchNorm1d(256),
#         #     nn.ReLU(inplace=True),
#         #     nn.Linear(256, output_dim),
#         #     nn.BatchNorm1d(output_dim, affine=False)  # output layer
#         # end 2 linears
        
#         self.head = nn.Sequential(  
#             nn.Linear(input_dim, 512)#,#256
#             # nn.BatchNorm1d(512),
#             # nn.ReLU(inplace=True),
#             # nn.Linear(256, output_dim),
#             # nn.BatchNorm1d(output_dim, affine=False),  # output layer
#             # nn.ReLU(inplace=True)  #remove
#         )
#     def forward(self, x):
#     # print("xxxx",x.shape)
#     # print("input_dim",input_dim)
#     # print("hidden_dim",hidden_dim)
#     # print("output_dim",output_dim)
#         return self.head (x)

class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim = 512):#: int = 512) -> None:
        super().__init__()
        # print("input_dim",input_dim)
        # print("hidden_dim",hidden_dim)
        # print("output_dim",output_dim)
        # self.head = nn.Sequential(
        #     # nn.Linear(input_dim, hidden_dim),
        #     # nn.BatchNorm1d(hidden_dim),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(input_dim, output_dim),
        #     nn.BatchNorm1d(output_dim, affine=False)  # output layer
        # )
        # 2 linears
        # self.head = nn.Sequential(  
        #     nn.Linear(input_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, output_dim),
        #     nn.BatchNorm1d(output_dim, affine=False)  # output layer
        # end 2 linears
        
        self.head = nn.Sequential(  
            nn.Linear(input_dim, 512),#,#256
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(output_dim, affine=False),  # output layer
            # nn.ReLU(inplace=True)  #remove
        )
    def forward(self, x):
    # print("xxxx",x.shape)
    # print("input_dim",input_dim)
    # print("hidden_dim",hidden_dim)
    # print("output_dim",output_dim)
        return self.head (x)

# class LocalEmbedding(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim) -> None:
#         super().__init__()
#         hidden_dim=512
#         output_dim=256
#         self.head = nn.Sequential(
#             nn.Conv1d(input_dim, hidden_dim,
#                       kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(hidden_dim, output_dim,
#                       kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm1d(output_dim)#, affine=False)  # output layer
#         )
#         # self.head = nn.Sequential(
#         #     nn.Conv3d(input_dim, hidden_dim,
#         #               kernel_size=1, stride=1, padding=0,bias=False),
#         #     nn.BatchNorm3d(hidden_dim),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv3d(hidden_dim, output_dim,
#         #               kernel_size=1, stride=1, padding=0,bias=False),
#         #     nn.BatchNorm3d(output_dim) #, affine=False # output layer
#         # )

#     def forward(self, x):
#         # print("permute",x.shape)
        
#         x = x.permute(0, 2, 1)
#         x = self.head (x )
        
#         # print("testttttt",x.shape,x.permute(0, 2, 1).shape)
#         return x.permute(0, 2, 1)

class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        hidden_dim=512
        output_dim=256
        # print("input_dim", input_dim,hidden_dim,output_dim)
        self.head = nn.Sequential(
            # nn.Conv1d(input_dim, hidden_dim,
            #           kernel_size=1, stride=1, padding=0),
            
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm1d(output_dim)#, affine=False)  # output layer
        )
        # self.head = nn.Sequential(
        #     nn.Conv3d(input_dim, hidden_dim,
        #               kernel_size=1, stride=1, padding=0,bias=False),
        #     nn.BatchNorm3d(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(hidden_dim, output_dim,
        #               kernel_size=1, stride=1, padding=0,bias=False),
        #     nn.BatchNorm3d(output_dim) #, affine=False # output layer
        # )

    def forward(self, x):
        # print("permute",x.shape)
        # print("locallll",x.shape)
        ##############################################33
        x = x.permute(0, 2, 1)
        x = self.head (x)
        #################################################
        # print("testttttt",x.shape,x.permute(0, 2, 1).shape)
        return x.permute(0, 2, 1)


class LocalEmbedding_image(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(output_dim)#, affine=False)  # output layer
        )
        # self.head = nn.Sequential(
        #     nn.Conv3d(input_dim, hidden_dim,
        #               kernel_size=1, stride=1, padding=0,bias=False),
        #     nn.BatchNorm3d(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(hidden_dim, output_dim,
        #               kernel_size=1, stride=1, padding=0,bias=False),
        #     nn.BatchNorm3d(output_dim) #, affine=False # output layer
        # )

    def forward(self, x):
        # print("permute",x.shape)
        
        x = x.permute(0,2,3,4,1)
        x = self.head(x)
        # print("testttttt",x.shape,x.permute(0, 2, 1).shape)
        

        return x#.permute(0, 4, 1,2,3)



class LocalEmbedding_3d(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.input_dim=256
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        output_dim=256
        print("hereeeee",input_dim,self.output_dim)
        self.head = nn.Sequential(
            nn.Conv3d(self.input_dim,hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm3d(output_dim)#, affine=False)  # output layer
            
        )
        # self.head = nn.Sequential(
        #     nn.Conv3d(input_dim, hidden_dim,
        #               kernel_size=1, stride=1, padding=0,bias=False),
        #     nn.BatchNorm3d(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(hidden_dim, output_dim,
        #               kernel_size=1, stride=1, padding=0,bias=False),
        #     nn.BatchNorm3d(output_dim) #, affine=False # output layer
        # )
        # self.conv1d_1=nn.Conv1d(4500,3000,#hidden_dim,
        #               kernel_size=1, stride=1, padding=0)
        # self.conv1d_2=nn.Conv1d(3000, 1158,#hidden_dim,
        #               kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        # print("x.shape",x.shape)
        # x = x.permute(0, 2,3,4,1)
        # print("localll_embedding",self.input_dim,self.hidden_dim,self.output_dim)
        x = self.head(x)
        # print("testttttt",x.shape,x.permute(0, 2, 1).shape)
        #commented out 1 line
        #############################################################################################3
        # x = rearrange(x, "b c w h l-> b   (w h l) c") 

        # x=self.conv1d_1(x)
        # x=self.conv1d_2(x)
        # print("tessstttt2",x.shape)
        return x#.permute(0, 4, 1,2,3)


# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()

#         # Define the key, query, and value linear transformations
#         self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
#         self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
#         self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

#         # Attention softmax
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         # Compute key, query, and value tensors
#         keys = self.key_conv(x)
#         queries = self.query_conv(x)
#         values = self.value_conv(x)

#         # Reshape keys, queries, and values
#         keys = keys.view(keys.size(0), keys.size(-1) * keys.size(-2) * keys.size(-3),-1)
#         queries = queries.view(queries.size(0), queries.size(-1) * queries.size(-2) * queries.size(-3),-1)
#         values = values.view(values.size(0), values.size(-1) * values.size(-2) * values.size(-3),-1)

#         # Compute attention scores
#         # print("qqq",queries.shape,keys.shape,values.shape)
#         attention = torch.bmm(keys.transpose(1, 2), queries)  #keys.transpose(1, 2)
#         attention = self.softmax(attention)

#         # Apply attention to values
#         # print("shapes",values.shape,attention.shape)
#         out = torch.bmm(values, attention.transpose(1, 2))

#         return out,attention


def attention_fn(query, context, temp1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Define the key, query, and value linear transformations

        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # Attention softmax
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.2)
        # self.bn1 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        # print("xxxx selfff", x.shape)
        # in_channels = x.shape[1]
        # self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()
        # self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()
        # self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()
        
        key = self.key_conv(x)
        key = self.dropout(key)
        # key = self.bn1(key)
        query = self.query_conv(x)
        query = self.dropout(query)
        # query = self.bn1(query)
        value = self.value_conv(x)
        value = self.dropout(value)
        # value = self.bn1(value)
        # key = x
        # query = x
        # value = x
        

        # Reshape keys, queries, and values
        # keys = keys.view(keys.size(0), keys.size(-1) * keys.size(-2) * keys.size(-3),-1)
        # queries = queries.view(queries.size(0), queries.size(-1) * queries.size(-2) * queries.size(-3),-1)
        # values = values.view(values.size(0), values.size(-1) * values.size(-2) * values.size(-3),-1)


        batch_size, channels,  height, width ,depth= x.size()
        # x=x.mean(1)
        # batch_size, height, width, depth = x.size()
 

        # query = query.view(batch_size, channels, -1)
        # key = key.view(batch_size, channels, -1)
        # value = value.view(batch_size, channels, -1)

        # query = query.view(batch_size, channels, -1)
        # key = key.view(batch_size, channels, -1)
        # value = value.view(batch_size, channels, -1)
        # query = query.view(batch_size,channels, -1, depth).mean(3)
        # # d_q = query.size(-1)
        # key = key.view(batch_size, channels, -1,depth).mean(3)
        #$$$$$$$$$
        # query = query.permute(0,1,-1,2,3)
        # key = key.permute(0,1,-1,2,3)
        # value = value.permute(0,1,-1,2,3)
        ##$$$$$$$$$$
        # print("valueeeeee",value.shape)
# ### this part
        query = query.view(batch_size, channels,-1)#,depth, -1)
        
        key = key.view(batch_size, channels,-1)#,depth, -1)
        # value = value.view(batch_size, depth*channels, -1)
  ### to here      
        # Compute attention scores
        # print("qqq",queries.shape,keys.shape,values.shape)
        # print("queryyyy",query.shape,key.shape)
        # mask_pred = torch.ones(batch_size, height*width*channels, depth).cuda()
        # here
        value = value.reshape(batch_size, channels,-1)#*depth, -1)
        
        # to here
        # key = self.key_conv(key)
        # query = self.query_conv(query)
        
        # value = self.value_conv(value)
        # value = value.view(batch_size, channels,depth, -1)
        # value = value.reshape(batch_size, channels*depth, -1)
        #$$$$
        # query = query.mean(2)
        # key = key.mean(2)
        #$$$$
        # value = value.reshape(batch_size, channels*depth, -1)
        attention = torch.bmm(query.permute(0, 2, 1), key)#/np.sqrt(d_q)  #keys.transpose(1, 2)
        # attention = nn.ReLU()(attention)
        attention = self.softmax(attention)

        # attention = self.dropout(attention)
        
        # Apply attention to values
        # print("shapes",values.shape,attention.shape)
        # print("value",value.shape)
        # print("value",value.shape,attention.shape)
        # out = torch.matmul(attention, value.permute(0, 2, 1))
        out = torch.bmm(value, attention.permute(0, 2, 1))

        # print("selfffffffffff",out.shape)
        
        out = out.view(batch_size, channels,height, width, depth)#  depth,height, width).permute(0,1,3,4,2)

        # atten_mask = mask_pred.view(batch_size, depth*channels, height*width)
        # final_mask = torch.matmul(attention, atten_mask.permute(0, 2, 1))
        # final_mask = final_mask.view(batch_size, channels,  height, width,depth)
        # final_mask = out - value.view(batch_size, channels,  height, width,depth)
        # print("finalllll", final_mask.shape)
        # out = out.view(batch_size, height, width,depth)
        temp = out
        # Residual connection and scaling
        out = self.gamma * out+ x
        # print("outtt",out.shape,attention.shape)
        # removed
        # print("outttttttttttt1",out.shape)
        # removed this line
        # out = out.reshape(batch_size,channels,  height*width*depth)
        # print("outttttttttttt1",out.shape)
        # out = out.view(batch_size, height* width,depth,channels)
        out = out.view(batch_size,channels,  height, width,depth)
        # out = out.view(batch_size,channels,  height*width*depth)
        # out = out.mean(2)
        # removed
        # out=out.mean(2)
        
        return out,attention#.float()#temp.mean(1)#temp.mean(1)#attention



        
class DepthAttention(nn.Module):
    def __init__(self, in_channels):
        super(DepthAttention, self).__init__()

        # Define the key, query, and value linear transformations
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # Attention softmax
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm3d(in_channels)
    def forward(self, x):
        # Compute key, query, and value tensors
        # print("x_aten",x.shape)
        
        key = self.key_conv(x)
        key = self.dropout(key)
        # key = self.bn1(key)
        query = self.query_conv(x)
        query = self.dropout(query)
        # query = self.bn1(query)
        value = self.value_conv(x)
        value = self.dropout(value)
        # value = self.bn1(value)
        # Reshape keys, queries, and values
        # keys = keys.view(keys.size(0), keys.size(-1) * keys.size(-2) * keys.size(-3),-1)
        # queries = queries.view(queries.size(0), queries.size(-1) * queries.size(-2) * queries.size(-3),-1)
        # values = values.view(values.size(0), values.size(-1) * values.size(-2) * values.size(-3),-1)


        batch_size, channels,  height, width ,depth= x.size()
        # x=x.mean(1)
        # batch_size, height, width, depth = x.size()
 

        # query = query.view(batch_size, channels, -1)
        # key = key.view(batch_size, channels, -1)
        # value = value.view(batch_size, channels, -1)

        # query = query.view(batch_size, channels, -1)
        # key = key.view(batch_size, channels, -1)
        # value = value.view(batch_size, channels, -1)

        # query = query.view(batch_size, height*width*channels, -1)
        # key = key.view(batch_size, height*width*channels, -1)

        query = x.view(batch_size, channels,height*width,-1)
        key = x.view(batch_size, channels,height*width,-1)
        
        # query = query.permute(0,1,-1,2)
        # key = key.permute(0,1,-1,2)
        # query = self.query_conv(query)
        # key = self.key_conv(key)
        query = query.mean(2)
        # d_q = query.size(-1)
        key = key.mean(2)
        
        # query = nn.AdaptiveAvgPool3d((4,512,1,5))(query).squeeze(3)
        # key = nn.AdaptiveAvgPool3d((4,512,1,5))(key).squeeze(3)
        # print("quey_key",query.shape,key.shape)

        # query = F.avg_pool3d(query, kernel_size=2, stride=2, padding=0)
        # key = F.max_pool3d(key, kernel_size=2, stride=2, padding=0)
        # query = (nn.AdaptiveAvgPool3d((512,1,5))(query)+nn.AdaptiveMaxPool3d((512,1,5))(query)).squeeze(2)
        # key = (nn.AdaptiveAvgPool3d((512,1,5))(key)+nn.AdaptiveMaxPool3d((512,1,5))(key)).squeeze(2)
        # print("quey_key",query.shape,key.shape)
        # value = value.view(batch_size, channels,-1)
        
        # value = value.view(batch_size, channels,height*width, -1)
        # value = self.value_conv(value)
        value = value.view(batch_size, channels*height*width, -1)
        # pred_mask = torch.ones(batch_size, height*width*channels, depth).cuda()
        # Compute attention scores
        # print("qqq",queries.shape,keys.shape,values.shape)
        # print("queryyyy",query.shape,key.shape)

        # key = self.key_conv(key)
        # query = self.query_conv(query)
        
        attention = torch.bmm(query.permute(0, 2, 1), key)#/np.sqrt(d_q)  #keys.transpose(1, 2)
        # attention = nn.ReLU()(attention)
        attention = self.softmax(attention)

        attention = self.dropout(attention)

        # Apply attention to values
        # print("shapes",value.shape,attention.shape)
        # print("value",value.shape)
        # print("value",value.shape,attention.shape)
        # out = torch.matmul(attention, value.permute(0, 2, 1))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        # print("depthhhhhhhhhhhhhhhh",out.shape)
        out = out.view(batch_size, channels,  height, width,depth)
        # atten_mask = torch.matmul(attention, pred_mask.permute(0, 2, 1))
        # atten_mask = atten_mask.view(batch_size, channels,  height, width,depth)
        # print("atten_mask",atten_mask.shape)
        # out = out.view(batch_size, height, width,depth)
        temp = out
        # Residual connection and scaling
        out = self.gamma * out + x
        # print("outtt",out.shape,attention.shape)

        # out = out.view(batch_size, depth* height* width,channels)

        # out = out.view(batch_size, height* width,depth,channels)
        # out=out.mean(1)
        # print("HII",out.shape,attention.shape)
        
        return out#,attention.float()#temp.mean(1)#temp.mean(1)#attention





class SelfAttention2(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Define the key, query, and value linear transformations
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # Attention softmax
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.25)
    def forward(self, x):
        # Compute key, query, and value tensors
        key = self.key_conv(x)
        query = self.query_conv(x)
        value = self.value_conv(x)

        # Reshape keys, queries, and values
        # keys = keys.view(keys.size(0), keys.size(-1) * keys.size(-2) * keys.size(-3),-1)
        # queries = queries.view(queries.size(0), queries.size(-1) * queries.size(-2) * queries.size(-3),-1)
        # values = values.view(values.size(0), values.size(-1) * values.size(-2) * values.size(-3),-1)
        batch_size, channels, depth, height, width = x.size()
        # query = query.view(batch_size, channels, -1)
        # key = key.view(batch_size, channels, -1)
        # value = value.view(batch_size, channels, -1)


        # Compute attention scores
        # print("qqq",queries.shape,keys.shape,values.shape)
        # attention = torch.matmul(query.permute(0, 2, 1), key)  #keys.transpose(1, 2)
        print("query,key",query.shape,key.shape)
        attention = torch.matmul(query, key.permute(0, 2,3,4,1))
        attention = self.softmax(attention)

        attention = self.dropout(attention)

        # Apply attention to values
        # print("shapes",values.shape,attention.shape)
        # print("value",value.shape)
        out = torch.matmul(attention, value.permute(0,2,3,4,1))#.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height, width)

        # Residual connection and scaling
        out = self.gamma * out + x
        out = out.view(batch_size, depth* height* width,channels)
        out=out.mean(1)
        # print("HII",out.shape,attention.shape)
        return out,attention
class ResNet_attention2(nn.Module):

    def __init__(self,use_attention, 
                 block,
                 layers,
                 block_inplanes,
                 model_depth,output_dim,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='A',
                 widen_factor=1,#0.5,  #1.0
                 n_classes=400,mode="pretraining"):
        super().__init__()
        # self.attention_layer = nn.MultiheadAttention(embed_dim=240*240, num_heads=1)#, dropout=0.1)
        ####
        # self.self_attention = SelfAttention(512)
        self.mode = mode
        self.text_feat_dim: int = 768
        # self.output_dim: int = 768,
        self.output_dim=output_dim
        self.hidden_dim: int = 256#2048
        self.interm_feature_dim=256#1024
        # self.depth_attention = DepthAttention(512)
        self.feature_dim = 512
        ####
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.use_attention = use_attention
        self.name = f"ResNet_pLGG_Classifer_depth{model_depth}"

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        # self.conv1 = nn.Conv3d(n_input_channels,
        #                        self.in_planes,
        #                        kernel_size=(conv1_t_size, 7, 7),
        #                        stride=(conv1_t_stride, 2, 2),
        #                        padding=(conv1_t_size // 2, 3, 3),
        #                        bias=False)
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.dropout_rate=0.25#0.1 #0.15
        self.dropout = nn.Dropout(self.dropout_rate)

        # self.global_embed = GlobalEmbedding(
        #         self.feature_dim, self.hidden_dim, self.output_dim
        #     )

        # self.local_embed = LocalEmbedding_3d(
        #     self.interm_feature_dim, self.hidden_dim, self.output_dim
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        # self.in_planes = planes * block.expansion  
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion  #was here
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.mode=="pretraining":
            return self.forward_pretraining(x)
        elif self.mode=="downstream":
            return self.forward_downstream(x)

#     def forward_pretraining(self, x):
#         # print("initttt",x.shape)
#         # b_size=x.shape[0]
#         # # x=x.permute(4, 0, 1, 2, 3)
#         # #x=x.view(b_size,240*240,155)
#         # x=x.view(b_size,155,240*240)
#         # x, _ = self.attention_layer(x, x, x)
#         # print("xx",x.shape)
#         # x=x.view(b_size,1,)
#         # x = self.dropout (x)
#         x = self.conv1 (x)
        
#         x = self.bn1 (x)
#         x = self.relu (x)
#         # if not self.no_max_pool:
#         x = self.maxpool (x)
#         x = self.dropout (x)
#         x = self.layer1 (x)
#         x = self.dropout(x)
#         x = self.layer2 (x)
#         x = self.dropout (x)
#         x = self.layer3 (x)
#         x = self.dropout (x)
#         # print("local_features",x.shape)
#         # print("localll",x.shape)
#         local_features = x
        
#         # print("local_featrues",local_features.shape)
#         x = self.layer4 (x)
#         # x = self.dropout (x)

#         x = self.avgpool (x)
        
#         x = x.view(x.size(0), -1)  #####changed for 3d
#         # print("feature_dim",x.shape)
#         # x = self.dropout (x)   #should be included probably
#         # x = self.fc(x)
#  #       x = torch.sigmoid(x)#softmax(x)
#         # local_features = rearrange(local_features, "b c w h l-> b (w h l) c")  ****????
#         # local_features = rearrange(local_features, "b c w h l-> b w h l c")
#         # print("contiguous",local_features.shape,local_features.contiguous().shape)
#         # print("globall",x.shape)
#         # print("imggg",x.shape)
#         return x,local_features.contiguous()
    def forward_pretraining(self, x):
        # print("initttt",x.shape)
        # b_size=x.shape[0]
        # # x=x.permute(4, 0, 1, 2, 3)
        # #x=x.view(b_size,240*240,155)
        # x=x.view(b_size,155,240*240)
        # x, _ = self.attention_layer(x, x, x)
        # print("xx",x.shape)
        # x=x.view(b_size,1,)
        # x = self.dropout (x)
        
        x = self.conv1 (x)
        
        x = self.bn1 (x)
        x = self.relu (x)
        # if not self.no_max_pool:
        x = self.maxpool (x)
        x = self.dropout (x)
        x = self.layer1 (x)
        x = self.dropout(x)
        x = self.layer2 (x)
        x = self.dropout (x)
        x = self.layer3 (x)
        # print("XXXXXXXX11",x.shape)
        x = self.dropout (x)
        # print("local_features",x.shape)
        # print("localll",x.shape)
        # print("XXXXX",x.shape)
        
        local_features = x# self.soft_attention(x)
        # print("x3",x.shape)
        # print("local_featrues",local_features.shape)
        x = self.layer4 (x)
        # local_features = x
        # print("x4",x.shape)
        # x = self.dropout (x)
        # x  =  self.depth_attention(x)





        x , _= self.self_attention(x)





        # print("x5",x.shape)
        x = self.avgpool (x)
        # print("hey1",x.shape)
        # x=x.mean(1)
        # print("hey",x.shape)
        # local_features = x
        # print("XXXXXXXX22",x.shape)
        x = x.view(x.size(0), -1)  #####changed for 3d
        # print("feature_dim",x.shape)
        # x = self.dropout (x)   #should be included probably
        # x = self.fc(x)
 #       x = torch.sigmoid(x)#softmax(x)
        # local_features = rearrange(local_features, "b c w h l-> b (w h l) c")  ****????
        # local_features = rearrange(local_features, "b c w h l-> b w h l c")
        # print("contiguous",local_features.shape,local_features.contiguous().shape)
        # print("globall",x.shape)
        # print("imggg",x.shape)

        # print("local_features",x)
        return x,local_features.contiguous()

    def forward_downstream(self, x):
        # print("initttt",x.shape)
        # b_size=x.shape[0]
        # # x=x.permute(4, 0, 1, 2, 3)
        # #x=x.view(b_size,240*240,155)
        # x=x.view(b_size,155,240*240)
        # x, _ = self.attention_layer(x, x, x)
        # print("xx",x.shape)
        # x=x.view(b_size,1,)
        # x = self.dropout (x)
        
        x = self.conv1 (x)
        
        x = self.bn1 (x)
        x = self.relu (x)
        # if not self.no_max_pool:
        x = self.maxpool (x)
        x = self.dropout (x)
        
        x = self.layer1 (x)
        x = self.dropout(x)
        
        x = self.layer2 (x)
        x = self.dropout (x)
        
        x = self.layer3 (x)
        x = self.dropout (x)
        # x  =  self.depth_attention2(x)
        # x = self.self_attention2(x)
        
        x = self.layer4 (x)
        
        # x = self.dropout (x)

        # x = self.avgpool (x)
        
        # x = x.view(x.size(0), -1)  #####changed for 3d
        # print("feature_dim",x.shape)
        # print("xxxx",x.shape)
        # x = x.view(x.size(0), x.size(2),-1)


        # x=x.permute(0,2,3,4,1)  ###******



        # print("HH",x.shape)
        # print("X",x.shape)





        # x=x.reshape(x.size(0),-1,x.size(-1))




        # x=x.reshape(x.size(0),x.size(1),-1)  ###*****
        # print("XXX@",x.shape)




        if self.use_attention==True:
            #newly added
            # x = x.mean(1)


            #newly added end
            # x  =  self.depth_attention(x)
            
            x ,atten_weights= self.self_attention(x)#(x,x,x,need_weights=True)  #, atten_weights
            # print("whyyyy",atten_weights.shape,atten_weights2.shape)
            # print("avg_pool",x.shape)

            x=self.avgpool (x)
            
            x = x.view(x.size(0), -1) 
            
            return x,atten_weights
        else:
            x=self.avgpool (x)
            # print("xxxxx",x.shape)
            x=x.view(x.size(0),-1)
            # x = torch.flatten(x, 1)
            return x
        # # print("X2",x.shape)
        # # x=
        # # x=x.reshape(x.size(0),8,8,5,512)
        # # x=x.permute(0,-1,1,2,3)
        # # x = self.avgpool (x)
        # # x = x.view(x.size(0), -1)  #####changed for 3d

        # # print("x",x.shape)
        # # Global average pooling
        # # x = self.avgpool (x)

        # x = x.mean(1)  ##$$$$
        # x = self.avgpool (x)
        
        # x = x.view(x.size(0), -1)  #####changed for 3d

        # print("SS",x.shape)
        # x = x.view(x.size(0), -1) 
        # x = x.mean(4).mean(3).mean(2)
        
        # x=x.reshape(x.shape[0], 8, 8, 5, 512)
        # x=x.permute(0,-1,1,2,3)
        # x=torch.mean(x, dim=1)
        

# class ResNet_attention(nn.Module):
#     def __init__(self,
#                  block,
#                  layers,
#                  output_dim=512,
#                  n_input_channels=1,
#                  conv1_t_size=7,
#                  conv1_t_stride=1,
#                  no_max_pool=False,
#                  shortcut_type='B',
#                  widen_factor=1,#0.5,  #1.0
#                  n_classes=400):
#         super(ResNet_attention, self).__init__()
#         self.inplanes = 64
#         self.text_feat_dim: int = 768
#         # self.output_dim: int = 768,
#         self.output_dim=output_dim
#         self.hidden_dim: int = 256#2048
#         self.interm_feature_dim=256#1024
#         ###????
#         self.feature_dim = 512
        
#         self.conv1 = nn.Conv3d(
#             1,
#             64,
#             kernel_size=7,
#             stride=(2, 2, 2),
#             padding=(3, 3, 3),
#             bias=False)
            
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
#         self.layer2 = self._make_layer(
#             block, 128, layers[1], shortcut_type, stride=2)
#         self.layer3 = self._make_layer(
#             block, 256, layers[2], shortcut_type, stride=1, dilation=2)
#         self.layer4 = self._make_layer(
#             block, 512, layers[3], shortcut_type, stride=1, dilation=4)

#         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         # self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
#         dropout_rate=0.25#0.1 #0.15
#         self.dropout = nn.Dropout(dropout_rate)

#         self.global_embed = GlobalEmbedding(
#                 self.feature_dim, self.hidden_dim, self.output_dim
#             )

#         self.local_embed = LocalEmbedding_3d(
#             self.interm_feature_dim, self.hidden_dim, self.output_dim
#         )

#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv3d):
#         #         m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
#         #     elif isinstance(m, nn.BatchNorm3d):
#         #         m.weight.data.fill_(1)
#         #         m.bias.data.zero_()
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight,
#                                         mode='fan_out',
#                                         nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(
#                     downsample_basic_block,
#                     planes=planes * block.expansion,
#                     stride=stride,
#                     no_cuda=self.no_cuda)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(
#                         self.inplanes,
#                         planes * block.expansion,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False), nn.BatchNorm3d(planes * block.expansion))

#         layers = []
#         layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, dilation=dilation))

#         return nn.Sequential(*layers)

#     def forward(self, x):
        
#         x = self.conv1 (x)
        
#         x = self.bn1 (x)
#         x = self.relu (x)
#         # if not self.no_max_pool:
#         x = self.maxpool (x)
#         x = self.dropout (x)
#         x = self.layer1 (x)
#         x = self.dropout(x)
#         x = self.layer2 (x)
#         x = self.dropout (x)
#         x = self.layer3 (x)
#         x = self.dropout (x)
#         # print("local_features",x.shape)
#         # print("localll",x.shape)
#         local_features = x
        
#         # print("local_featrues",local_features.shape)
#         x = self.layer4 (x)
#         # x = self.dropout (x)
        
#         x = self.avgpool (x)
        
#         x = x.view(x.size(0), -1)  #####changed for 3d
#         # print("feature_dim",x.shape)
#         # x = self.dropout (x)   #should be included probably
#         # x = self.fc(x)
#  #       x = torch.sigmoid(x)#softmax(x)
#         # local_features = rearrange(local_features, "b c w h l-> b (w h l) c")  ****????
#         # local_features = rearrange(local_features, "b c w h l-> b w h l c")
#         # print("contiguous",local_features.shape,local_features.contiguous().shape)
#         # print("globall",x.shape)
#         # print("imggg",x.shape)
#         return x,local_features.contiguous()
        

#     def forward(self, x):
#         # print("initttt",x.shape)
        
#         x = self.dropout (x)
#         x = self.conv1 (x)
#         x = self.bn1 (x)
#         x = self.relu (x)
#         if not self.no_max_pool:
#             x = self.maxpool (x)

#         x = self.layer1 (x)
#         x = self.dropout(x)
#         x = self.layer2 (x)
#         x = self.dropout (x)
#         x = self.layer3 (x)
#         x = self.dropout (x)
#         # print("local_features",x.shape)
#         # print("localll",x.shape)
#         local_features = x
#         # print("local_featrues",local_features.shape)
#         x = self.layer4 (x)
#         # x = self.dropout (x)

#         x = self.avgpool (x)
        
#         x = x.view(x.size(0), -1)  #####changed for 3d
#         # print("feature_dim",x.shape)
#         # x = self.dropout (x)   #should be included probably
#         # x = self.fc(x)
#  #       x = torch.sigmoid(x)#softmax(x)
#         # local_features = rearrange(local_features, "b c w h l-> b (w h l) c")  ****????
#         # local_features = rearrange(local_features, "b c w h l-> b w h l c")
#         # print("contiguous",local_features.shape,local_features.contiguous().shape)
#         # print("globall",x.shape)
#         # print("imggg",x.shape)
#         return x,local_features.contiguous()


# transformers
class BertClassifier_attention(nn.Module): #(GPT2ForSequenceClassification):#

    def __init__(self,output_dim, freeze_bert = True,dropout=0.25):

        super(BertClassifier_attention, self).__init__()


        self.tokenizer = LongformerTokenizerFast.from_pretrained("yikuan8/Clinical-Longformer")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        
        self.bert_layer = LongformerForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer",
                                                        gradient_checkpointing=False,
                                                        attention_window = 512 ,output_attentions=True,output_hidden_states=True)

        self.embedding_dim: int = 768
        # self.output_dim: int = 768,
        self.output_dim=output_dim
        self.hidden_dim: int = 256
        self.last_n_layers = 1

        self.global_embed = GlobalEmbedding(
            self.embedding_dim)#, self.hidden_dim, self.output_dim)
        self.local_embed = LocalEmbedding(
            self.embedding_dim, self.hidden_dim, self.output_dim)

       
        
        self.bert_layer.config.pad_token_id = self.bert_layer.config.eos_token_id#############################333
        config = LongformerConfig()
       
        self.bert_layer.classifier.out_proj=nn.Linear(768, 768) #256
        
        for param in self.bert_layer.parameters():
            
            param.requires_grad = False
        for param in self.bert_layer.longformer.encoder.layer[11].parameters():#score #pooler
            param.requires_grad = True
        for param in self.bert_layer.classifier.parameters():#score #pooler
            param.requires_grad = True
        

        self.dropout1 = nn.Dropout(dropout)
        
        self.fc1=nn.Linear(1024, 512)
        self.fc3=nn.Linear(512,256)
        
        self.fc2=nn.Linear(256, 1)
        self.linear_layer = nn.Linear(949, 768)


    # def aggregate_tokens(self, embeddings, text_ids,last_layer_attn):
    #     '''
    #     :param embeddings: bz, 1, 112, 768Ss
    #     :param caption_ids: bz, 112
    #     :param last_layer_attn: bz, 111
    #     '''
    #     _,num_layers, num_words, dim = embeddings.shape
    #     embeddings = embeddings.permute(0, 2, 1, 3)
    #     agg_embs_batch = []
    #     sentences = []
    #     last_attns = []

    #     # loop over batch
    #     for embs,text_id, last_attn in zip(embeddings, text_ids,last_layer_attn):
    #         agg_embs = []
    #         token_bank = []
    #         words = []
    #         word_bank = []
    #         attns = []
    #         attn_bank = []
            
    #         # loop over sentence
    #         for word_emb, word_id, attn in zip(embs, text_id,last_attn):  #######????
    #             word = self.idxtoword[word_id.item()]
    #             # print("wordddd", word)
    #             # print("word",word)
    #             # print("word",word,attn)
    #             if word == "</s>":# or word=="<s>":#"[ELECTRA]":#"[sep]":#"[SEP]":    #<s>
    #                 new_emb = torch.stack(token_bank)
    #                 new_emb = new_emb.sum(axis=0)
    #                 agg_embs.append(new_emb)
    #                 words.append("".join(word_bank))
    #                 attns.append(sum(attn_bank))
    #                 agg_embs.append(word_emb)
    #                 words.append(word)
    #                 attns.append(attn)
    #                 break
    #             # This is because some words are divided into two words.
    #             if not word.startswith("Ġ"):  #Ċ   #Ġ
    #                 if len(word_bank) == 0:
    #                     token_bank.append(word_emb)
    #                     word_bank.append(word)
    #                     attn_bank.append(attn)
    #                 else:
    #                     new_emb = torch.stack(token_bank)
    #                     new_emb = new_emb.sum(axis=0) #axis
    #                     agg_embs.append(new_emb)
    #                     words.append("".join(word_bank))
                        
    #                     attns.append(sum(attn_bank))

    #                     token_bank = [word_emb]
    #                     word_bank = [word]
    #                     attn_bank = [attn]
    #             else:
    #                 token_bank.append(word_emb)
    #                 word_bank.append(word[1:])
    #                 attn_bank.append(attn)
    #         # print("attention bank",attns)
    #             # print("worddd",word,attn)
    #         agg_embs = torch.stack(agg_embs)
    #         padding_size = num_words - len(agg_embs)
    #         paddings = torch.zeros(padding_size, num_layers, dim)
    #         # paddings = paddings.type_as(agg_embs)
    #         words = words + ['<pad>'] * padding_size  #"[PAD]"
    #         last_attns.append(
    #             torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
    #         agg_embs_batch.append(torch.cat([agg_embs, paddings]))
    #         sentences.append(words)

    #     agg_embs_batch = torch.stack(agg_embs_batch)
    #     agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
    #     last_atten_pt = torch.stack(last_attns)
    #     last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
    #     # print("sentence",agg_embs_batch.shape)
    #     # print("last_atten_pt",last_atten_pt.nonzero().squeeze())
    #     print("sentence",sentences)
    #     print("agg_embs_batch",agg_embs_batch)
    #     return agg_embs_batch, sentences, last_atten_pt

    def aggregate_tokens2(self, embeddings, text_ids,last_layer_attn):
        '''
        :param embeddings: bz, 1, 112, 768
        :param caption_ids: bz, 112
        :param last_layer_attn: bz, 111
        '''
        _,num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []
        # print("embbb",embeddings)
        # loop over batch
        for embs,text_id, last_attn in zip(embeddings, text_ids,last_layer_attn):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            attns = []
            attn_bank = []
            
            # loop over sentence
            for word_emb, word_id, attn in zip(embs, text_id,last_attn):  #######????
                word = self.idxtoword[word_id.item()]
                # print("wordddd", word)
                # print("word",word)
                # print("word",word,attn)
                if word == "</s>":#"[ELECTRA]":#"[sep]":#"[SEP]":    #<s>
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    break
                # This is because some words are divided into two words.
                if not word.startswith("Ġ"):  #Ċ   #Ġ
                    if len(word_bank) == 0:
                        token_bank=[word_emb]
                        word_bank=[word]
                        attn_bank=[attn]
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        
                        attns.append(sum(attn_bank))

                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                else:
                    token_bank=[word_emb]
                    word_bank=[word[1:]]
                    attn_bank=[attn]
            # print("attention bank",attns)
                # print("worddd",word,attn)
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.type_as(agg_embs)
            words = words + ['<pad>'] * padding_size  #"[PAD]"
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
        # print("sentence",agg_embs_batch.shape)
        # print("last_atten_pt",last_atten_pt.nonzero().squeeze())
        # print("sentence",sentences)
        # print("agg_embs_batch",agg_embs_batch)
        return agg_embs_batch, sentences, last_atten_pt
        
        

    def forward(self, input_ids, attention_mask = None):
        attention_mask = (input_ids != 0).float() 
        bert_l=self.bert_layer(input_ids= input_ids, attention_mask=attention_mask,return_dict=True)
        # print("bert_l",self.bert_layer)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == 0] = 0
        pooled_output=bert_l.logits#######
        #self.bert_layer.longformer.encoder.layer[11]
        dropout_output = self.dropout1(pooled_output)
        # print("atttttt",bert_l)#.attentions)
        #[:, -1, :, 1:]
        # print("hhhh",bert_l.attentions[-1].shape)
        # print("attention",bert_l.attentions[11])
        last_layer_attn = bert_l.attentions[-1][:, :, 1:, 0].mean(dim=1)#[:, :, 0, 1:].mean(dim=1)
        # print("shape",last_layer_attn.shape)
        # print("last_layer_attn1",bert_l.attentions[-1][:, -1, :, 1:].mean(dim=-1))#last_layer_attn[:, :, 0, 1:])#[:, :, 0, 1:])#.mean(dim=1))
        # print("last_layer_attn2",last_layer_attn.mean(dim=1))
        # print("bert_l_attention",bert_l.attentions[-1].nonzero().squeeze())#[:, :, 0, 1:].mean(dim=1).nonzero().squeeze())
        all_feat = bert_l.hidden_states[-1].unsqueeze(1)#.hidden_states[-1].unsqueeze(1)  #.last_hidden_state #hidden_states[11]
        
        test2=bert_l.hidden_states[-1]
        # print("test",test2.shape)
        # print(all_feat)
        # print("all_feat",all_feat.shape)

        
        all_feat2=all_feat
        all_feat, sents, last_atten_pt = self.aggregate_tokens2(all_feat, input_ids, last_layer_attn)
        # print("last_atten_pt",last_atten_pt.nonzero().squeeze())
        last_atten_pt = last_atten_pt[:, 1:].contiguous()
        # print("all_feat",all_feat.shape)
        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0]
        report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()
        word_feat=word_feat.squeeze(1)
        # print("all_feat",report_feat.shape,word_feat.shape)
        # all_feat2=all_feat2.view(all_feat2.size(0),-1,768)
        # all_feat2=all_feat2[:, 0, :]  this one global
        # all_feat2 = torch.mean(all_feat2, dim=1)  this one global
        # print("all_feat2",all_feat2.shape)
        # print("report_feat",report_feat.shape)
        # all_feat2=self.linear_layer(all_feat2)
        all_feat2 = pooled_output
        # print("pooled",all_feat2.shape)


        #remove special tokens
        self.mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>") |
                             (np.array(sents)[:, 1:] == '<s>') |
                             (np.array(sents)[:, 1:] == '</s>') |
                             (np.array(sents)[:, 1:] == '<mask>') |
                             (np.array(sents)[:, 1:] == 'unk') |
                             (np.array(sents)[:, 1:] == '') |
                             (np.array(sents)[:, 1:] == 'Ċ')).type(torch.bool)

# Create a mask to filter tensor1
        mask = ~self.mask_pad.unsqueeze(-1)  # Expand mask to match tensor1's shape

# Filter tensor1 based on the mask
        
        word_feat = word_feat[mask]
        sents = sents[mask]
        # self.mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>")|(np.array(sents)[:, 1:] =='<s>')|(np.array(sents)[:, 1:] =='</s>')|(np.array(sents)[:, 1:] =='<mask>')|(np.array(sents)[:, 1:] =='unk')|(np.array(sents)[:, 1:] =='')|(np.array(sents)[:, 1:] =='Ċ')).type_as(img_embed).bool()
        # self.text_masks = torch.tensor([token in self.mask_pad for token in text_tokens], dtype=torch.bool)
        return  all_feat,all_feat2, word_feat, last_atten_pt, sents#report_feat, word_feat, last_atten_pt, sents





class BertClassifier_global_local(nn.Module): #(GPT2ForSequenceClassification):#

    def __init__(self,output_dim, freeze_bert = True,dropout=0.25):

        super(BertClassifier_global_local, self).__init__()


        self.tokenizer = LongformerTokenizerFast.from_pretrained("yikuan8/Clinical-Longformer")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        
        # self.bert_layer = LongformerForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer",
        #                                                 gradient_checkpointing=False,
        #                                                 attention_window = 512 ,output_attentions=True,output_hidden_states=True)

        self.bert_layer = LongformerModel.from_pretrained("yikuan8/Clinical-Longformer",
                                                        gradient_checkpointing=False,
                                                        attention_window = 512 ,output_attentions=True,output_hidden_states=True)
        self.embedding_dim: int = 768
        # self.output_dim: int = 768,
        self.output_dim=output_dim
        self.hidden_dim: int = 256
        self.last_n_layers = 1

        self.global_embed = GlobalEmbedding(
            self.embedding_dim)#, self.hidden_dim, self.output_dim)
        self.local_embed = LocalEmbedding(
            self.embedding_dim, self.hidden_dim, self.output_dim)

       
        
        # self.bert_layer.config.pad_token_id = self.bert_layer.config.eos_token_id#############################333
        config = LongformerConfig()
       
        # self.bert_layer.classifier.out_proj=nn.Linear(768, 768) #256
        
        for param in self.bert_layer.parameters():
            
            param.requires_grad = False
        for param in self.bert_layer.encoder.layer[11].parameters():#score #pooler
            param.requires_grad = True
        for param in self.bert_layer.encoder.layer[10].parameters():#score #pooler
            param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[9].parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[8].parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[7].parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.classifier.parameters():#score #pooler
        #     param.requires_grad = True
        

        self.dropout1 = nn.Dropout(dropout)
        
        # self.fc1=nn.Linear(1024, 512)
        # self.fc3=nn.Linear(512,256)
        
        # self.fc2=nn.Linear(256, 1)
        # self.linear_layer = nn.Linear(949, 768)


    def aggregate_tokens3(self, embeddings, text_ids,input_attention):
        '''
        :param embeddings: bz, 1, 112, 768
        :param caption_ids: bz, 112
        :param last_layer_attn: bz, 111
        '''

        
        _,num_layers, num_words, dim = embeddings.shape
        # print("shappe",embeddings.shape)
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []
        # loop over batch
        for embs,text_id,atten in zip(embeddings, text_ids,input_attention):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            attns = []
            attn_bank = []
            
            # print("embbbbbbbbb1",embs.shape,text_ids.shape)
            # loop over sentence
            for word_emb, word_id,att in zip(embs, text_id,atten):  #######????
                # print("embbbbbbbbb2",word_emb.shape,word_id.shape)
                word = self.idxtoword[word_id.item()]
                # print("wordddd", word)
                # print("hereeeeeeeeeeeeeeeeeeeeeeeeeeee",word)
                # if "Ċ" in word:
                #     print("firstttttttttttt",word)
                if word == "</s>":
                    new_emb = torch.stack(token_bank)
                    # print("new_emb",new_emb.shape)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    
                    agg_embs.append(word_emb)
                    words.append(word)
                    
                    attns.append(sum(attn_bank))
                    attns.append(att)
                    
                    break
                # This is because some words are divided into two words.
                if not word.startswith("Ġ"):  
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(att)
                    else:
                        if not word.startswith("Ċ"):
                            token_bank.append(word_emb)
                            word_bank.append(word)
                            attn_bank.append(att)
                        else:
                            if len(word)>1:
                                new_emb = torch.stack(token_bank)
                                new_emb = new_emb.sum(axis=0)
                                agg_embs.append(new_emb)
                                words.append("".join(word_bank))
                                attns.append(sum(attn_bank))

                                token_bank=[word_emb]
                                word_bank=[word[1:]]
                                attn_bank = [att]
                else:
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))

                    token_bank=[word_emb]
                    word_bank=[word[1:]]
                    attn_bank = [att]
            
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            # paddings = paddings.type_as(agg_embs)
            paddings = paddings.to(agg_embs.device)
            words = words + ['<pad>'] * padding_size  #"[PAD]"
            
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)
            #### commented for special tokens
        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
             #### commented for special tokens
            
        # last_atten_pt = torch.stack(last_attns)
        # last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
        # print("sentence",agg_embs_batch.shape)
        # print("last_atten_pt",last_atten_pt.nonzero().squeeze())
        # print("sentence",sentences)
        # print("agg_embs_batch",agg_embs_batch)
        # print("sentences",sentences)

        # special_tokens = ['<s>', '</s>', '<pad>','<mask>' , 'unk']
        # filtered_tokens = []
        # filtered_word_embeddings = []

        # for token, embedding in zip(sentences, agg_embs_batch):
        #     if token not in special_tokens:
        #         filtered_tokens.append(token)
        #         filtered_word_embeddings.append(embedding)
        # filtered_word_embeddings = torch.stack(filtered_word_embeddings)
        # filtered_word_embeddings = filtered_word_embeddings.permute(0, 2, 1, 3)
        # print("tokennnnnss",filtered_tokens)

        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
        # print("sentences",sentences,"Ċ") ""
        for sentence in sentences:
            for word in sentence:
                if "Ċ" in word:
                    print(word)
        return agg_embs_batch, sentences, last_atten_pt

    def aggregate_tokens2(self, embeddings, text_ids,input_attention):
        '''
        :param embeddings: bz, 1, 112, 768
        :param caption_ids: bz, 112
        :param last_layer_attn: bz, 111
        '''
        _,num_layers, num_words, dim = embeddings.shape
        # print("shappe",embeddings.shape)
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []
        # loop over batch
        for embs,text_id,atten in zip(embeddings, text_ids,input_attention):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            attns = []
            attn_bank = []
            
            # print("embbbbbbbbb1",embs.shape,text_ids.shape)
            # loop over sentence
            for word_emb, word_id,att in zip(embs, text_id,atten):  #######????
                # print("embbbbbbbbb2",word_emb.shape,word_id.shape)
                word = self.idxtoword[word_id.item()]
                # print("wordddd", word)
                # print("hereeeeeeeeeeeeeeeeeeeeeeeeeeee",word)
                if word == "</s>":
                    new_emb = torch.stack(token_bank)
                    # print("new_emb",new_emb.shape)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    
                    agg_embs.append(word_emb)
                    words.append(word)
                    
                    attns.append(sum(attn_bank))
                    attns.append(att)
                    
                    break
                if word == "Ċ":
                    continue
                # This is because some words are divided into two words.
                if not word.startswith("Ġ"):  
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(att)
                    else:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(att)
                else:
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))

                    token_bank=[word_emb]
                    word_bank=[word[1:]]
                    attn_bank = [att]
            
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            # paddings = paddings.type_as(agg_embs)
            paddings = paddings.to(agg_embs.device)
            words = words + ['<pad>'] * padding_size  #"[PAD]"
            
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)
            #### commented for special tokens
        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
             #### commented for special tokens
            
        # last_atten_pt = torch.stack(last_attns)
        # last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
        # print("sentence",agg_embs_batch.shape)
        # print("last_atten_pt",last_atten_pt.nonzero().squeeze())
        # print("sentence",sentences)
        # print("agg_embs_batch",agg_embs_batch)
        # print("sentences",sentences)

        # special_tokens = ['<s>', '</s>', '<pad>','<mask>' , 'unk']
        # filtered_tokens = []
        # filtered_word_embeddings = []

        # for token, embedding in zip(sentences, agg_embs_batch):
        #     if token not in special_tokens:
        #         filtered_tokens.append(token)
        #         filtered_word_embeddings.append(embedding)
        # filtered_word_embeddings = torch.stack(filtered_word_embeddings)
        # filtered_word_embeddings = filtered_word_embeddings.permute(0, 2, 1, 3)
        # print("tokennnnnss",filtered_tokens)

        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
        # print("sentences",sentences,"Ċ") ""
        # for sentence in sentences:
        #     for word in sentence:
        #         if "Ċ" in word:
        #             print(word)
        return agg_embs_batch, sentences, last_atten_pt
        
        

    def forward(self, input_ids, attention_mask = None):
        
        
        # global_attention_mask = None#(input_ids != 0).float()

        # global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.bool).cuda()
        # global_attention_mask[0] = True

        # global_attention_mask = torch.ones(input_ids.shape, dtype=torch.long).cuda()
        bert_l=self.bert_layer(input_ids= input_ids, attention_mask=attention_mask,return_dict=True,output_attentions=True)#,global_attention_mask=global_attention_mask)
        # print("attention",bert_l.attentions[-1][:,:,1:,:].shape)
        word_attention = bert_l["attentions"][-1] [:,:,1:,:].mean(dim=1).mean(dim=-1)
        # print("atttttttttttttttt", bert_l.global_attentions[-1].shape)#bert_l["global_attention"])
        # print("heyyyy",input_ids['token_type_ids'])
        # print("new_sentence_embeddingggggggggggggggggggg",bert_l.hidden_states[-1][0][input_ids['token_type_ids'][0] == 1])
        # print("bert_l",self.bert_layer)
        # attention_mask = torch.ones_like(input_ids)  #################333
        # attention_mask[input_ids == 0] = 0   ##################3333333
        # pooled_output=bert_l#.logits#######
        # #self.bert_layer.longformer.encoder.layer[11]
        # dropout_output = self.dropout1(pooled_output)
        
        # print("atttttt",bert_l)#.attentions)
        #[:, -1, :, 1:]
        # print("hhhh",bert_l.attentions[-1].shape)
        # print("attention",bert_l.attentions[11])
        # last_layer_attn = bert_l.attentions[-1][:, :, 1:, 0].mean(dim=1)#[:, :, 0, 1:].mean(dim=1) ###########################333
        # print("shape",last_layer_attn.shape)
        # print("last_layer_attn1",bert_l.attentions[-1][:, -1, :, 1:].mean(dim=-1))#last_layer_attn[:, :, 0, 1:])#[:, :, 0, 1:])#.mean(dim=1))
        # print("last_layer_attn2",last_layer_attn.mean(dim=1))
        # print("bert_l_attention",bert_l.attentions[-1].nonzero().squeeze())#[:, :, 0, 1:].mean(dim=1).nonzero().squeeze())
        # all_feat = bert_l.hidden_states[-1].unsqueeze(1)#.hidden_states[-1].unsqueeze(1)  #.last_hidden_state #hidden_states[11]
        all_feat = bert_l.last_hidden_state.unsqueeze(1)#
        cls1=all_feat
        # print("alll_feat",all_feat.shape)
        
        ###### new
        # print("allllll 1",all_feat.shape)
        # all_feat = all_feat.permute(1, 0, 2, 3)   # new????????
        # print("allllll 2",all_feat.shape)
        word_emb, sents , w_atten = self.aggregate_tokens2(all_feat, input_ids,word_attention)

        w_atten = w_atten[:, 1:].contiguous()
        # print("sens",word_emb.shape, len(sents[0]))

        # special_tokens = ['<s>', '</s>', '<pad>','<mask>' , 'unk','','Ċ']
        # filtered_tokens = []
        # filtered_word_embeddings = []

        # for tokens, embedding in zip(sents, word_emb):
        #     print("tokennhh",embedding.shape,len(tokens))
        #     for i in range(len(tokens)):
        #         if tokens[i] not in special_tokens:
        #         # print("hhhhh",token[0])
        #             filtered_tokens.append(tokens[i])
        #             filtered_word_embeddings.append(embedding[0][i][:])
        # print("tokennhh",len(filtered_word_embeddings),filtered_word_embeddings[0].shape)
        # filtered_word_embeddings = torch.stack(filtered_word_embeddings)
        # print("tokennnnnss",filtered_tokens)

        # filtered_word_embeddings = filtered_word_embeddings.permute(0, 2, 1, 3)
        # print("tokennnnnss",len(filtered_tokens),filtered_tokens[0].shape)
        word_embeddings = word_emb.mean(axis=1)
        batch_dim, num_words, feat_dim = word_embeddings.shape
        
        word_feat=word_embeddings[:,1:,:]
        # report_feat=word_embeddings[:,0,:]
        report_feat=torch.mean(word_embeddings[:,1:,:],dim=1)
        # word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        
        # word_embeddings = self.local_embed(word_feat)
        # print("attention layer",word_embeddings.shape)
        word_feat = word_feat.view(batch_dim, num_words-1, self.embedding_dim)
        # word_embeddings = word_embeddings.permute(0, 2, 1)

        

        
        # word_embeddings = word_embeddings / torch.norm(
        #     word_embeddings, 2, dim=1, keepdim=True
        # ).expand_as(word_embeddings)
        

        # ##### new
        
        # # if self.last_n_layers == 1:
        # #     all_feat = all_feat[:, 0]
        # # report_feat = all_feat[:, 0].contiguous()
        # # word_feat = all_feat[:, 1:].contiguous()
        # # word_feat=word_feat.squeeze(1)
       
        # # all_feat2=all_feat2.view(all_feat2.size(0),-1,768)
        # cls1=cls1[:, 0, :] # this one global   # new global
        # # all_feat2 = torch.mean(all_feat2, dim=1)  this one global
        # cls1=cls1[:,0,:]    # new global
        # # print("all_feat2",all_feat2.shape)
        # # print("report_feat",report_feat.shape)
        # # all_feat2=self.linear_layer(all_feat2)
        
        # # all_feat2 = pooled_output
        # # print("cls",cls1.shape,all_feat2.shape)
        # all_feat2=cls1
        # # print("pooled",all_feat2.shape)
        # report_feat=report_feat[:, 0]
        # print("insideee",report_feat.shape,word_feat.shape)
        # print("shape",report_feat.shape,word_feat.shape)
        #remove special tokens
        # self.mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>") |
        #                      (np.array(sents)[:, 1:] == '<s>') |
        #                      (np.array(sents)[:, 1:] == '</s>') |
        #                      (np.array(sents)[:, 1:] == '<mask>') |
        #                      (np.array(sents)[:, 1:] == 'unk') |
        #                      (np.array(sents)[:, 1:] == '') |
        #                      (np.array(sents)[:, 1:] == 'Ċ')).type(torch.bool)

# Create a mask to filter tensor1
        # print("sents",len(sents),len(sents[0]),sents[0][0])
        # sents = sents((sents[:, 1:] == "<pad>") |
        #                      (np.array(sents)[:, 1:] == '<s>') |
        #                      (np.array(sents)[:, 1:] == '</s>') |
        #                      (np.array(sents)[:, 1:] == '<mask>') |
        #                      (np.array(sents)[:, 1:] == 'unk') |
        #                      (np.array(sents)[:, 1:] == '') |
        #                      (np.array(sents)[:, 1:] == 'Ċ'))
#         mask = ~self.mask_pad.unsqueeze(-1)  # Expand mask to match tensor1's shape

# # Filter tensor1 based on the mask
        
        
#         expanded_mask = mask.expand(-1, -1, word_feat.size(-1))
#         num_remaining_elements = expanded_mask.sum(dim=1).squeeze()
#         print("word_feat",expanded_mask)
#         num_remaining_elements = expanded_mask.sum(dim=1).squeeze()
#         word_feat = word_feat[expanded_mask].view(word_feat.shape[0], num_remaining_elements, 768)
#         print("word_feat",word_feat.shape)
       
        return  report_feat,word_feat,sents,w_atten#all_feat,all_feat2, word_feat,sents#report_feat, word_feat, last_atten_pt, sents

#(4,256,30,15,10)
#(4,949,768)

# transformers
class BertClassifier_global(nn.Module): #(GPT2ForSequenceClassification):#

    def __init__(self,output_dim, freeze_bert = True,dropout=0.25):

        super(BertClassifier_global, self).__init__()


        self.tokenizer = LongformerTokenizerFast.from_pretrained("yikuan8/Clinical-Longformer")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        
        # self.bert_layer = LongformerForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer",
        #                                                 gradient_checkpointing=False,
        #                                                 attention_window = 512 ,output_attentions=True,output_hidden_states=True)
        self.bert_layer = LongformerModel.from_pretrained("yikuan8/Clinical-Longformer",
                                                        gradient_checkpointing=False,
                                                        attention_window = 512 ,output_attentions=True,output_hidden_states=True)
        # self.bert_layer = LongformerForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer",
        #                                                 gradient_checkpointing=False,
        #                                                 attention_window = 512 ,output_attentions=True,output_hidden_states=True)
        
        # print("MODEL",[name for name, _ in self.bert_layer.named_children()])
        self.embedding_dim: int = 768
        # self.output_dim: int = 768,
        self.output_dim=output_dim
        self.hidden_dim: int = 256
        self.last_n_layers = 1

        self.global_embed = GlobalEmbedding(
            self.embedding_dim, self.hidden_dim, self.output_dim)
        

       
        
        # self.bert_layer.config.pad_token_id = self.bert_layer.config.eos_token_id#############################333
        config = LongformerConfig()
       
        # self.bert_layer.classifier.out_proj=nn.Linear(768, 768) #256  #768
        
        for param in self.bert_layer.parameters():
            
            param.requires_grad = False
        for param in self.bert_layer.encoder.layer[11].parameters():#score #pooler
            param.requires_grad = True
        for param in self.bert_layer.encoder.layer[10].parameters():#score #pooler
            param.requires_grad = True
        for param in self.bert_layer.encoder.layer[9].parameters():#score #pooler
            param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[8].parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[7].parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[6].parameters():#score #pooler
        #     param.requires_grad = True
        
        # for param in self.bert_layer.encoder.layer[5].parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[4].parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[3].parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[2].parameters():#score #pooler
        #     param.requires_grad = True
        
        # for param in self.bert_layer.longformer.encoder.layer[9].parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.classifier.parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.pooler.parameters():#score #pooler  #lm_head
        #     param.requires_grad = True
        # for param in self.bert_layer.embeddings.parameters():#score #pooler
        #     param.requires_grad = True
        

        self.dropout1 = nn.Dropout(dropout)
        
        # self.fc1=nn.Linear(1024, 512)
        # self.fc3=nn.Linear(512,256)
        
        # self.fc2=nn.Linear(256, 1)
        # self.linear_layer = nn.Linear(949, 768)

    def aggregate_tokens2(self, embeddings, text_ids):
        '''
        :param embeddings: bz, 1, 112, 768
        :param caption_ids: bz, 112
        :param last_layer_attn: bz, 111
        '''
        # _,num_layers, num_words, dim = embeddings.shape
        _, num_words, dim = embeddings.shape
        # print("shappe",embeddings.shape)
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        
        # loop over batch
        for embs,text_id in zip(embeddings, text_ids):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            
            # print("embbbbbbbbb1",embs.shape,text_ids.shape)
            # loop over sentence
            for word_emb, word_id in zip(embs, text_id):  #######????
                # print("embbbbbbbbb2",word_emb.shape,word_id.shape)
                word = self.idxtoword[word_id.item()]
                # print("wordddd", word)
                
                if word == "</s>":
                    new_emb = torch.stack(token_bank)
                    # print("new_emb",new_emb.shape)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    
                    agg_embs.append(word_emb)
                    words.append(word)
                    
                    break
                # This is because some words are divided into two words.
                if not word.startswith("Ġ"):  
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        
                    else:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        
                else:
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    token_bank=[word_emb]
                    word_bank=[word[1:]]
                    
            
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            # paddings = paddings.type_as(agg_embs)
            paddings = paddings.to(agg_embs.device)
            words = words + ['<pad>'] * padding_size  #"[PAD]"
            
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        # last_atten_pt = torch.stack(last_attns)
        # last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
        # print("sentence",agg_embs_batch.shape)
        # print("last_atten_pt",last_atten_pt.nonzero().squeeze())
        # print("sentence",sentences)
        # print("agg_embs_batch",agg_embs_batch)
        # print("sentences",sentences)
        return agg_embs_batch, sentences#, last_atten_pt
        
        

    def forward(self, input_ids, attention_mask = None):
        bert_l=self.bert_layer(input_ids= input_ids, attention_mask=attention_mask,return_dict=True)
        
        # pooled_output=bert_l.logits
        # dropout_output = self.dropout1(pooled_output)
        # all_feat2 = dropout_output
        # print("bert_lll",self.bert_layer)
        # all_feat = bert_l.hidden_states[-1].unsqueeze(1)#.hidden_states[-1].unsqueeze(1)  #.last_hidden_state #hidden_states[11]
        

        all_feat=bert_l.last_hidden_state
        # all_feat=bert_l.hidden_states[-1]

        # cls1=all_feat
        # print("now shape",all_feat.shape)
        # print("cls1",all_feat[:,0,0,:])
        # print("alll_feat",all_feat.shape)
        
        ###### new
        # print("allllll 1",all_feat.shape)
        # all_feat = all_feat.permute(1, 0, 2, 3)   # new????????
        # print("allllll 2",all_feat.shape)
        
        # new_all_feat=torch.mean(all_feat[:,0,:,:],dim=1)
        
        # new_all_feat=torch.mean(all_feat,dim=1)
        # all_feat, sents= self.aggregate_tokens2(all_feat, input_ids)  ####**************************************
        

        ##### new
         
        # if self.last_n_layers == 1:  ************************************
        #     all_feat = all_feat[:, 0] **********************
        
        # print("sentences",sents[0])
        # all_feat2=all_feat2.view(all_feat2.size(0),-1,768)
        # print("now clsss", all_feat.shape)
        # all_feat=all_feat[:, 0, :] # this one global   # new global
        all_feat=torch.mean(all_feat[:,1:,:],dim=1)
        # all_feat=all_feat[:,1:,:].mean(dim=1)
        # print("cls2",all_feat)
        # print("all_feat,", all_feat.shape)
        # all_feat2 = torch.mean(all_feat2, dim=1)  this one global
        
        
        return  all_feat

class image_text_attention_global(torch.nn.Module):
    def __init__(self,emb_dim,num_heads,mode):
        super().__init__()


        self.emb_dim=emb_dim
        self.num_heads=num_heads
        inplanes=[64, 128, 256, 512]
        self.cnn=ResNet_attention2(BasicBlock,[1,1,1,1],model_depth=18, n_classes=1039,block_inplanes=inplanes,output_dim=self.emb_dim)	
        self.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
		# net.fc = net.fc = nn.Linear(512, 1)#3) #512		
        # Below lines are for sentence embedding which has size 150 for self_trained_embedding
        if mode=="global_local":
            self.transformer=BertClassifier_global_local(output_dim=self.emb_dim) #BertClassifier_attention
        else:
            self.transformer=BertClassifier_global(output_dim=self.emb_dim)
        
        # Adding intermediate linear layer before final classification
        num_intermediate_output = 512
        # self.intermediate_layer = nn.Linear(1024, num_intermediate_output) #256
        self.intermediate_layer = nn.Linear(1024, num_intermediate_output) 
        # Adding a final classification layer that takes in outputs 3
        self.final_classification_layer = nn.Linear(num_intermediate_output, 1)

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, batch_first=True)

		
    def forward(self, image, input_ids, attention_mask = None):
        # print("batch1 started")		                    
        img_feat_q, patch_feat_q  = self.cnn(image) # ****
        # img_feat_q= self.cnn(image) 
        # print("path",patch_feat_q.shape) 
        patch_emb_q = patch_feat_q#self.cnn.local_embed(patch_feat_q) # ****
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)   #****
        # # patch_emb_q = rearrange(patch_emb_q, "b  w h l c-> b  (w h l) c")
        # # patch_emb_q = rearrange(patch_emb_q, "b  w c h l -> b   (w h l) c")  ########****
        # # print("nextt,",patch_emb_q.shape)
        # img_emb_q = self.cnn.global_embed(img_feat_q)  # removed
        img_emb_q = img_feat_q
        # print("img_emb_q",img_emb_q.shape)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
                
        # Pass sentence embedding through linear layers
        # all_features,report_feat_q, word_feat_q, word_attn_q, sents = self.transformer(input_ids, attention_mask) *****
        report_feat_q= self.transformer(input_ids, attention_mask)

        # print("report_feat_q",report_feat_q.shape)

        # word_emb_q = self.transformer.local_embed(word_feat_q)   *****
        # word_emb_q = F.normalize(word_emb_q, dim=-1)  *****
        report_emb_q = self.transformer.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        



        combined = torch.cat((img_emb_q, report_emb_q), axis=1)
        
        intermediate = self.intermediate_layer(combined)
        # intermediate = self.intermediate_layer(img_emb_q)
        
        # Get final labels
        labels = self.final_classification_layer(intermediate)

        # mask_pad = torch.from_numpy(np.array(sents)[:, 1:] == "<pad>").type_as(image).bool()  #[PAD]"      ***
        # print("PPPP",patch_emb_q.shape)
        # patch_atten_output, _ = self.patch_local_atten_layer(patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask_pad,need_weights=True)   ****
        # patch_atten_output = F.normalize(patch_atten_output, dim=-1)   ****
        # word_atten_output, _ = self.word_local_atten_layer(word_emb_q, patch_emb_q, patch_emb_q,need_weights=True)   ****
        # word_atten_output = F.normalize(word_atten_output, dim=-1)  ****


        return  labels,img_emb_q,report_emb_q,patch_emb_q#,word_emb_q,word_attn_q,sents,patch_atten_output,word_atten_output#,image_global_attention,report_global_attention


class image_text_attention(torch.nn.Module):
    def __init__(self,emb_dim=256,num_heads=2,mode="global_local"):
        super().__init__()


        self.emb_dim=256#emb_dim
        self.num_heads=1#num_heads
        self.mode=mode
        inplanes=[64, 128, 256, 512]
        self.cnn=ResNet_attention2(False,BasicBlock,[2,2,2,2],model_depth=18, n_classes=1039,block_inplanes=inplanes,output_dim=self.emb_dim,mode="pretraining")	
        self.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
		#resnet18()#
        # net.fc = net.fc = nn.Linear(512, 1)#3) #512		
        # Below lines are for sentence embedding which has size 150 for self_trained_embedding
        self.global_embed = GlobalEmbedding()
        if self.mode=="global_local":
            self.transformer=BertClassifier_global_local(output_dim=self.emb_dim) #BertClassifier_attention
        else:
            self.transformer=BertClassifier_global(output_dim=self.emb_dim)
        
        
        self.patch_local_atten_layer = CrossAttention(512,"img_txt")#,sents)
            
        self.word_local_atten_layer = CrossAttention(512,"txt_img")#,sents)
        
        # Adding intermediate linear layer before final classification
        # num_intermediate_output = 64
        # self.intermediate_layer = nn.Linear(1024, num_intermediate_output) #256

        # # Adding a final classification layer that takes in outputs 3
        # self.final_classification_layer = nn.Linear(num_intermediate_output, 1)

        # patch local attention layer
        # self.patch_local_atten_layer = CrossAttention(512,"image_text")








        # self.patch_local_atten_layer = nn.MultiheadAttention(self.emb_dim, self.num_heads, batch_first=True)
        # # # # sentence local attention layer
        # # self.word_local_atten_layer = CrossAttention(512,"text_image")#
        # self.word_local_atten_layer = nn.MultiheadAttention(self.emb_dim, self.num_heads, batch_first=True)

		
    def forward(self, image, input_ids, attention_mask = None):
        # print("batch1 started")		                    
        img_feat_q, patch_feat_q  = self.cnn(image)#,mode="pretraining") # ****
        # img_feat_q= self.cnn(image) 
        # print("path",patch_feat_q.shape) 
        # patch_emb_q = self.cnn.local_embed(patch_feat_q)
        patch_emb_q = patch_feat_q# ****
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)   #****
        # print("path",patch_feat_q.shape) 
        # # patch_emb_q = rearrange(patch_emb_q, "b  w h l c-> b  (w h l) c")
        # patch_emb_q = rearrange(patch_emb_q, "b c w h l -> b c (w h l) ")
        # # patch_emb_q = rearrange(patch_emb_q, "b  w c h l -> b   (w h l) c")  ########****
        # # print("nextt,",patch_emb_q.shape)
        img_emb_q = self.global_embed(img_feat_q)
        # img_emb_q = img_feat_q
        # print("img_emb_q",img_emb_q.shape)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
                
        # Pass sentence embedding through linear layers
        report_feat_q, word_feat_q, sents ,merged_att = self.transformer(input_ids, attention_mask) #*****
        # word_feat_q,sents= self.transformer(input_ids, attention_mask)

        # print("img_emb_q",img_emb_q.shape)
        
        # word_emb_q = self.transformer.local_embed(word_feat_q)
        word_emb_q = word_feat_q  # *****
        word_emb_q = F.normalize(word_emb_q, dim=-1)#  *****
        # print("all_words",word_emb_q.shape,patch_emb_q.shape)
        # print("attention layer",word_emb_q.shape,patch_emb_q.shape)
        # print("nn.multihead",torch.bmm(word_emb_q,patch_emb_q.permute(0, 2, 1)).shape)
        # attention=nn.MultiheadAttention(128, num_heads=1, batch_first=True)
        # a_layer,_=attention(word_emb_q,patch_emb_q,patch_emb_q)
        report_emb_q = self.transformer.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        



        # combined = torch.cat((img_emb_q, report_emb_q), axis=1)
        # intermediate = self.intermediate_layer(combined)
        # # intermediate = self.intermediate_layer(img_emb_q)
        
        # # Get final labels
        # labels = self.final_classification_layer(intermediate)
         
        # Convert the padding mask to a tensor
        padding_mask_list=['<s>', '</s>', '<pad>','<mask>' , 'unk','','Ċ']
        # mask_pad = torch.from_numpy(np.array(sents)[:, 1:] == "<pad>").type_as(image).bool()  #[PAD]"      ***
        # mask_pad = torch.from_numpy(np.array(sents)[:, 1:] == any(item == 2 for item in list1)).type_as(image).bool()  #[PAD]"      ***
        # print("PPPP",patch_emb_q.shape)
        # print("patch",patch_emb_q.shape,word_emb_q.shape)
        #########################33
        # mask_pad = torch.from_numpy(np.any(sents[:, 1:][:, np.newaxis] == np.array(padding_mask_list)[np.newaxis, :], axis=-1)).type(torch.bool)
        mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>")|(np.array(sents)[:, 1:] =='<s>')|(np.array(sents)[:, 1:] =='</s>')|(np.array(sents)[:, 1:] =='<mask>')|(np.array(sents)[:, 1:] =='unk')|(np.array(sents)[:, 1:] =='')|(np.array(sents)[:, 1:] =='Ċ')).type_as(image).bool()
# Printing the mask (True where the condition is met, False otherwise)
        # print("mask_pred",mask_pad)
       
###########################################
        # with torch.no_grad():
        #     bz = report_emb_q.shape[0]
        #     atten_weights = merged_att.detach()
            
        #     word_atten_weights = []

        #     for i in range(bz):
                
        #         atten_weight = atten_weights[i]

        #         nonzero = atten_weight.nonzero().squeeze()
        #         low = torch.quantile(atten_weight[nonzero], 0.1)
        #         high = torch.quantile(atten_weight[nonzero], 0.9)
        #         atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
        #         word_atten_weights.append(atten_weight.clone())
        #     word_atten_weights = torch.stack(word_atten_weights)
        #     # TODO: maybe clip the tensor of 10 percentile and 90 percentile

        # word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)
                    #############################
        # print("shapessss",word_emb_q.shape,word_atten_weights.shape)
        # print("tessttttt",(torch.sum(word_emb_q * word_atten_weights.unsqueeze(-1))).shape)
        # patch_atten_output, patch_weights = self.patch_local_atten_layer(patch_emb_q, word_emb_q * word_atten_weights.unsqueeze(-1), word_emb_q * word_atten_weights.unsqueeze(-1), key_padding_mask=mask_pad,need_weights=True)  # ****
        # print("ppppppppppppppppppppppppppppppppp",patch_emb_q.shape,word_emb_q.shape)
        ####deleted
        # patch_atten_output, patch_weights = self.patch_local_atten_layer(patch_emb_q, word_emb_q , word_emb_q, key_padding_mask=mask_pad,need_weights=True)  # ****
#######


        #############&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        # self.patch_local_atten_layer = CrossAttention(512,"img_txt",sents)
            
        # self.word_local_atten_layer = CrossAttention(512,"txt_img",sents)
        ##############&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        patch_atten_output, patch_weights, new_patch_emb_q = self.patch_local_atten_layer(patch_emb_q,word_emb_q,sents)#(patch_emb_q, word_emb_q , word_emb_q , key_padding_mask=mask_pad,need_weights=True) 
        patch_atten_output = F.normalize(patch_atten_output, dim=-1) #  ****

        
        # word_atten_output, word_weights = self.word_local_atten_layer(word_emb_q * word_atten_weights.unsqueeze(-1), patch_emb_q, patch_emb_q,need_weights=True) #  ****
        # word_atten_output, word_weights = self.word_local_atten_layer(word_emb_q, patch_emb_q, patch_emb_q,need_weights=True) #  ****

        



        word_atten_output, word_weights ,new_word_emb_q= self.word_local_atten_layer(patch_emb_q,word_emb_q,sents)# (word_emb_q , patch_emb_q, patch_emb_q,need_weights=True)
        # word_atten_output = F.normalize(word_atten_output, dim=-1) # ****
        # print("initialss",patch_emb_q.shape,word_emb_q.shape,new_patch_emb_q.shape,new_word_emb_q.shape,patch_atten_output.shape,word_atten_output.shape)
        word_atten_output = F.normalize(word_atten_output, dim=-1)
        # patch_atten_output = F.normalize(patch_atten_output, dim=-1)
        # word_emb_q = self.transformer.local_embed(word_feat_q)
        # word_emb_q = F.normalize(word_emb_q, dim=-1)
        # patch_emb_q = self.cnn.local_embed(patch_feat_q)
        # patch_emb_q = F.normalize(patch_emb_q, dim=-1)   #**** 
        
        patch_weights = F.normalize(patch_weights, dim=-1)
        word_weights = F.normalize(word_weights, dim=-1)
        # print("repr shapes",patch_weights.shape,word_weights.shape,word_atten_output.shape,patch_atten_output.shape,patch_emb_q.shape,word_emb_q.shape)
        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) -1 for sent in sents
        ]
        # print("HIIII SHAPE", patch_atten_output.shape,word_atten_output.shape,patch_weights.shape,word_weights.shape)
        # print("worddddd",word_atten_output.shape,word_weights.shape)
        # return  img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_weights,word_weights,cap_lens,patch_atten_output,word_atten_output,merged_att
        # print("shapppppeeess",new_patch_emb_q.shape,new_word_emb_q.shape, patch_atten_output.shape,word_atten_output.shape)
        print("dddddddddddddddd", img_emb_q.shape, report_emb_q.shape, new_patch_emb_q.shape, new_word_emb_q.shape)
        return  img_emb_q,report_emb_q,new_patch_emb_q,new_word_emb_q,sents,patch_weights,word_weights,cap_lens,patch_atten_output,word_atten_output,merged_att

# class downstream_image_classifier(torch.nn.Module):
#     def __init__(self,emb_dim,num_heads):
#         super().__init__()

#         inplanes=[64, 128, 256, 512]
#         self.emb_dim=emb_dim
#         self.cnn=ResNet_attention(BasicBlock,[1,1,1,1],model_depth=18, n_classes=1039,block_inplanes=inplanes,output_dim=self.emb_dim)	
#         self.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
#         for param in self.cnn.parameters():
#             param.requires_grad = False
        
        
#         num_intermediate_output = 64
#         self.intermediate_layer = nn.Linear(128, num_intermediate_output) #256

#         # Adding a final classification layer that takes in outputs 3
#         self.final_classification_layer = nn.Linear(num_intermediate_output, 1)

        

		
#     def forward(self, image):	                    
#         img_feat_q, patch_feat_q  = self.cnn(image)
#         img_emb_q = self.cnn.global_embed(img_feat_q)
#         img_emb_q = F.normalize(img_emb_q, dim=-1)

        
#         intermediate = self.intermediate_layer(img_emb_q)
        
#         labels = self.final_classification_layer(intermediate)

    
#         return  labels

class image_text_attention_prototype(torch.nn.Module):
    def __init__(self,emb_dim,num_heads):
        super().__init__()


        self.emb_dim=emb_dim
        self.num_heads=num_heads
        self.num_prototypes = 2
        self.freeze_prototypes_epochs= 1
        self.proto_temperature: float = 0.2
        inplanes=[64, 128, 256, 512]
        self.cnn=ResNet_attention(BasicBlock,[1,1,1,1],model_depth=18, n_classes=1039,block_inplanes=inplanes,output_dim=self.emb_dim)	
        self.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
		# net.fc = net.fc = nn.Linear(512, 1)#3) #512		
        # Below lines are for sentence embedding which has size 150 for self_trained_embedding
       
        self.transformer=BertClassifier_attention(output_dim=self.emb_dim)
        
        # Adding intermediate linear layer before final classification
        num_intermediate_output = 64
        self.intermediate_layer = nn.Linear(256, num_intermediate_output) #256

        # Adding a final classification layer that takes in outputs 3
        self.final_classification_layer = nn.Linear(num_intermediate_output, 1)

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, batch_first=True)

        self.prototype_layer = nn.Linear(self.emb_dim, self.num_prototypes, bias=False)
        self.sinkhorn_iterations: int = 3
        self.epsilon: float = 0.05

        # if self._use_ddp_or_dpp2(self.trainer):
        #     self.get_assignments = self.distributed_sinkhorn
        # else:
        self.get_assignments = self.sinkhorn
		
    def forward(self, image, input_ids, attention_mask = None):
        # print("batch1 started")		                    
        img_feat_q, patch_feat_q  = self.cnn(image)
        # print("path",patch_feat_q.shape)
        patch_emb_q = self.cnn.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        # patch_emb_q = rearrange(patch_emb_q, "b  w h l c-> b  (w h l) c")
        # patch_emb_q = rearrange(patch_emb_q, "b  w c h l -> b   (w h l) c")  ########****
        # print("nextt,",patch_emb_q.shape)
        img_emb_q = self.cnn.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
                
        # Pass sentence embedding through linear layers
        all_features,report_feat_q, word_feat_q, word_attn_q, sents = self.transformer(input_ids, attention_mask)
        # print("img_emb_q",img_emb_q.shape)

        word_emb_q = self.transformer.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.transformer.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        



        combined = torch.cat((img_emb_q, report_emb_q), axis=1)
        intermediate = self.intermediate_layer(combined)
        # intermediate = self.intermediate_layer(img_emb_q)
        
        # Get final labels
        labels = self.final_classification_layer(intermediate)

        mask_pad = torch.from_numpy(np.array(sents)[:, 1:] == "<pad>").type_as(image).bool()  #[PAD]"      
        # print("PPPP",patch_emb_q.shape)
        patch_atten_output, _ = self.patch_local_atten_layer(patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask_pad,need_weights=True)
        patch_atten_output = F.normalize(patch_atten_output, dim=-1)
        word_atten_output, _ = self.word_local_atten_layer(word_emb_q, patch_emb_q, patch_emb_q,need_weights=True)
        word_atten_output = F.normalize(word_atten_output, dim=-1)



        #prototype
        with torch.no_grad():
            w = self.prototype_layer.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototype_layer.weight.copy_(w)

        # Compute assign code of images
        img_proto_out = self.prototype_layer(img_emb_q)
        report_proto_out = self.prototype_layer(report_emb_q)

        # TODO: define this to hparams
        with torch.no_grad():
            img_code = torch.exp(
                img_proto_out / self.epsilon).t()
            img_code = self.get_assignments(
                img_code, self.sinkhorn_iterations)         # bz, 500
            report_code = torch.exp(
                report_proto_out / self.epsilon).t()
            report_code = self.get_assignments(
                report_code, self.sinkhorn_iterations)       # bz, 500

        img_proto_prob = F.softmax(
            img_proto_out / self.proto_temperature, dim=1)
        report_proto_prob = F.softmax(
            report_proto_out / self.proto_temperature, dim=1)



        return  labels,img_emb_q,report_emb_q,patch_emb_q,word_emb_q,word_attn_q,sents,patch_atten_output,word_atten_output#,img_code,report_code,img_proto_prob,report_proto_prob

    def sinkhorn(self, Q, nmb_iters):
        ''' 
            :param Q: (num_prototypes, batch size)
        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            # if self.hparams.gpus > 0:
            u = torch.zeros(K).cuda()
            r = torch.ones(K).cuda() / K
            c = torch.ones(B).cuda() / B
            # else:
            #     u = torch.zeros(K)
            #     r = torch.ones(K) / K
            #     c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()







class downstream_image_classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        inplanes=[64, 128, 256, 512]
        self.emb_dim=512
        self.cnn=ResNet_attention(BasicBlock,[2,2,2,2],model_depth=18, n_classes=1039,block_inplanes=inplanes,output_dim=self.emb_dim)	
        self.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.cnn.fc = nn.Linear(512, 1)
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        
        num_intermediate_output = 64
        self.intermediate_layer = nn.Linear(128, num_intermediate_output) #256

        # Adding a final classification layer that takes in outputs 3
        # self.final_classification_layer = nn.Linear(num_intermediate_output, 1)
        self.gap = nn.AdaptiveAvgPool2d(1) # gap = GlobalAveragePooling
        self.relu = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(512, 256)
        self.do = nn.Dropout(0.5)
        self.projection_region = nn.Linear(self.cnn.output_dim, 128)
        self.projection_global = nn.Linear(self.cnn.output_dim, 128)

        

		
    def forward(self, image):	                    
        img_feat_q, patch_feat_q  = self.cnn(image)
        img_emb_q = self.cnn.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
        patch_emb_q = self.cnn.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        print("embedds",img_emb_q.shape,patch_emb_q.shape)
        # g_p = torch.squeeze(self.gap(patch_emb_q)) # N, 512
        g_p = self.gap(patch_emb_q) 
        print("g_p",g_p.shape)
        f_r = self.relu(self.projection_region(self.do(g_p))) # N, 256
        f_g = self.relu(self.projection_global(self.do(img_emb_q))) # N, 256
        print("f_g",f_g.shape)
        net = torch.cat((f_r,f_g), dim=-1) #  N, 512 + 512 (1024 --> 512 --> 256 --> 14)

        
        # intermediate = self.intermediate_layer(net)
        
        # labels = self.final_classification_layer(intermediate)

        labels=self.cnn.fc(net)
        return  labels

class downstream_image_classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print("YES")
        inplanes=[64, 128, 256, 512]
        # inplanes=[128, 256, 512,1024]
        self.cnn=ResNet_attention2(True,BasicBlock,[2,2,2,2],model_depth=18,block_inplanes=inplanes,output_dim=512,n_classes=1039,mode="downstream")	
        # self.cnn = ResNet_attention(True, BasicBlock, [2, 2, 2, 2],mode="downstream")
        # self.cnn=ResNet_attention(BasicBlock,[2,2,2,2],model_depth=18, n_classes=1039,block_inplanes=inplanes)	

        self.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        # for param in self.cnn.parameters():
        #     param.requires_grad = False
        
        
        # num_intermediate_output = 128
        # self.intermediate_layer = nn.Linear(128, num_intermediate_output) #256

        # Adding a final classification layer that takes in outputs 3
        # self.fc = nn.Linear(num_intermediate_output, 1)el
        self.gap = nn.AdaptiveAvgPool2d(1) # gap = GlobalAveragePooling
        self.relu = nn.ReLU(inplace=True)
        

        # self.do = nn.Dropout(0.5)
        # self.projection_region = nn.Linear(self.cnn.output_dim, 128)
        # self.projection_global = nn.Linear(self.cnn.output_dim, 128)

        

		
    def forward(self, image):	
        image = image.float()   
        if self.cnn.use_attention:                 
            img_feat_q ,attn = self.cnn(image)
            labels  = self.cnn.fc1(self.cnn.fc(img_feat_q))
            
            return  labels,attn
        # img_emb_q = self.cnn.global_embed(img_feat_q)
        # img_emb_q = F.normalize(img_emb_q, dim=-1)
        # patch_emb_q = self.cnn.local_embed(patch_feat_q)
        # patch_emb_q = F.normalize(patch_emb_q, dim=-1)

        # g_p = torch.squeeze(self.gap(patch_emb_q)) # N, 512
        # f_r = self.relu(self.projection_region(self.do(g_p))) # N, 256
        # f_g = self.relu(self.projection_global(self.do(img_emb_q))) # N, 256
        # net = torch.cat((f_r,f_g), dim=-1) #  N, 512 + 512 (1024 --> 512 --> 256 --> 14)
        
        # x = self.dropout(x)

        # x = self.avgpool(x)

        # x = x.view(x.size(0), -1)
        # # x = self.fc(x)
        # intermediate = self.intermediate_layer(img_emb_q)
        
            
        # labels=self.cnn.fc(img_feat_q)
            
            
           #,attention_weight#(nn.Sigmoid()(labels)>= 0.5).squeeze().to(torch.long)#,attention_weight
        else:
            
            img_feat_q = self.cnn(image)
            # print("image_feat",img_feat_q.shape)
            labels = self.cnn.fc1(self.cnn.fc(img_feat_q))
            # print("labells",labels.shape)
            return  labels


class downstream_image_text_classifier(torch.nn.Module):
    def __init__(self,emb_dim,num_heads):
        super().__init__()

        inplanes=[64, 128, 256, 512]
        self.emb_dim=emb_dim
        self.cnn=ResNet_attention(True,BasicBlock,[1,1,1,1],model_depth=18, n_classes=1039,block_inplanes=inplanes,output_dim=self.emb_dim,mode="downstream")	
        self.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.transformer=BertClassifier_attention(output_dim=self.emb_dim)
        self.gap = nn.AdaptiveAvgPool2d(128) # gap = GlobalAveragePooling
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512, 256)
        self.do = nn.Dropout(0.1)
        
        self.projection_region = nn.Linear(self.cnn.output_dim, 64)
        self.projection_global = nn.Linear(self.cnn.output_dim, 64)
        # for param in self.cnn.parameters():
        #     param.requires_grad = False
        # for param in self.cnn.layer4.parameters():
        #     param.requires_grad = True
        
        # for param in self.transformer.parameters():
        #     param.requires_grad = False
        
        num_intermediate_output = 64#128
        self.intermediate_layer = nn.Linear(128 , num_intermediate_output) #256

        # Adding a final classification layer that takes in outputs 3
        self.final_classification_layer = nn.Linear(num_intermediate_output, 1)

        

		
    def forward(self, image, input_ids, attention_mask = None):	                    
        img_feat_q, patch_feat_q  = self.cnn(image)
        img_emb_q = self.cnn.global_embed(img_feat_q)
        # print(img_emb_q.shape)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
        patch_emb_q = self.cnn.local_embed(patch_feat_q)
        patch_emb_q = rearrange(patch_emb_q, "b c w h l-> b (w h l) c") ####(#(((#((#(#)))))))
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        g_p = torch.squeeze(self.gap(patch_emb_q)) # N, 512
        f_r = self.relu(self.projection_region(self.do(g_p))) # N, 256
        f_r = f_r.view(f_r.size(0),-1)
        f_g = self.relu(self.projection_global(self.do(img_emb_q))) # N, 256
        net = f_g#torch.cat((f_r,f_g), dim=-1)

        report_feat_q, word_feat_q, word_attn_q, sents = self.transformer(input_ids, attention_mask)
        # print("img_emb_q",img_emb_q.shape)

        word_emb_q = self.transformer.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.transformer.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        t_g_p = torch.squeeze(self.gap(word_emb_q)) # N, 512
        t_f_r = self.relu(self.projection_region(self.do(t_g_p))) # N, 256
        t_f_g = self.relu(self.projection_global(self.do(report_emb_q))) # N, 256
        t_f_r = t_f_r.view(f_r.size(0),-1)
        t_net = t_f_g#torch.cat((t_f_r,t_f_g), dim=-1) #  N, 512 + 512 (1024 --> 512 --> 256 --> 14)

        
        
        combined = torch.cat((net, t_net), axis=1)
        
        intermediate = self.intermediate_layer(combined)
        
        labels = self.final_classification_layer(intermediate)

    
        return  labels


















class image_text_attention_test(torch.nn.Module):
    def __init__(self,emb_dim,num_heads):
        super().__init__()


        self.emb_dim=emb_dim
        self.num_heads=num_heads
        self.num_prototypes = 2
        self.freeze_prototypes_epochs= 1
        self.proto_temperature: float = 0.2
        inplanes=[64, 128, 256, 512]
        self.cnn=ResNet_attention(BasicBlock,[1,1,1,1],model_depth=18, n_classes=1039,block_inplanes=inplanes,output_dim=self.emb_dim)	
        self.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
		# net.fc = net.fc = nn.Linear(512, 1)#3) #512		
        # Below lines are for sentence embedding which has size 150 for self_trained_embedding
       
        self.transformer=BertClassifier_attention(output_dim=self.emb_dim)
        
        # Adding intermediate linear layer before final classification
        num_intermediate_output = 64
        self.intermediate_layer = nn.Linear(256, num_intermediate_output) #256

        # Adding a final classification layer that takes in outputs 3
        self.final_classification_layer = nn.Linear(128, 1)

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, batch_first=True)

        self.prototype_layer = nn.Linear(self.emb_dim, self.num_prototypes, bias=False)
        self.sinkhorn_iterations: int = 3
        self.epsilon: float = 0.05

        # if self._use_ddp_or_dpp2(self.trainer):
        #     self.get_assignments = self.distributed_sinkhorn
        # else:
        self.get_assignments = self.sinkhorn
		
    def forward(self, image, input_ids, attention_mask = None):
        # print("batch1 started")		                    
        img_feat_q, patch_feat_q  = self.cnn(image)
        # print("path",patch_feat_q.shape)
        patch_emb_q = self.cnn.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        # patch_emb_q = rearrange(patch_emb_q, "b  w h l c-> b  (w h l) c")
        # patch_emb_q = rearrange(patch_emb_q, "b  w c h l -> b   (w h l) c")  ########****
        # print("nextt,",patch_emb_q.shape)
        img_emb_q = self.cnn.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
                
        # Pass sentence embedding through linear layers
        all_features,report_feat_q, word_feat_q, word_attn_q, sents = self.transformer(input_ids, attention_mask)
        # print("img_emb_q",img_emb_q.shape)
        # print("all_features_text",all_features.shape,report_feat_q.shape)
        # print("all_features_image",img_feat_q.shape)

        word_emb_q = self.transformer.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.transformer.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        



        combined = torch.cat((img_emb_q, report_emb_q), axis=1)
        intermediate = self.intermediate_layer(combined)
        # intermediate = self.intermediate_layer(img_emb_q)
        
        # Get final labels
        labels = self.final_classification_layer(intermediate)

        mask_pad = torch.from_numpy(np.array(sents)[:, 1:] == "<pad>").type_as(image).bool()  #[PAD]"      
        # print("PPPP",patch_emb_q.shape)
        patch_atten_output, _ = self.patch_local_atten_layer(patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask_pad,need_weights=True)
        patch_atten_output = F.normalize(patch_atten_output, dim=-1)
        word_atten_output, _ = self.word_local_atten_layer(word_emb_q, patch_emb_q, patch_emb_q,need_weights=True)
        word_atten_output = F.normalize(word_atten_output, dim=-1)



        #prototype
        # with torch.no_grad():
        #     w = self.prototype_layer.weight.data.clone()
        #     w = F.normalize(w, dim=1, p=2)
        #     self.prototype_layer.weight.copy_(w)

        # # Compute assign code of images
        # img_proto_out = self.prototype_layer(img_emb_q)
        # report_proto_out = self.prototype_layer(report_emb_q)

        # # TODO: define this to hparams
        # with torch.no_grad():
        #     img_code = torch.exp(
        #         img_proto_out / self.epsilon).t()
        #     img_code = self.get_assignments(
        #         img_code, self.sinkhorn_iterations)         # bz, 500
        #     report_code = torch.exp(
        #         report_proto_out / self.epsilon).t()
        #     report_code = self.get_assignments(
        #         report_code, self.sinkhorn_iterations)       # bz, 500

        # img_proto_prob = F.softmax(
        #     img_proto_out / self.proto_temperature, dim=1)
        # report_proto_prob = F.softmax(
        #     report_proto_out / self.proto_temperature, dim=1)



        return  labels,img_emb_q,report_emb_q,patch_emb_q,word_emb_q,word_attn_q,sents,patch_atten_output,word_atten_output#,img_code,report_code,img_proto_prob,report_proto_prob

    def sinkhorn(self, Q, nmb_iters):
        ''' 
            :param Q: (num_prototypes, batch size)
        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            # if self.hparams.gpus > 0:
            u = torch.zeros(K).cuda()
            r = torch.ones(K).cuda() / K
            c = torch.ones(B).cuda() / B
            # else:
            #     u = torch.zeros(K)
            #     r = torch.ones(K) / K
            #     c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()










class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Define the key, query, and value linear transformations
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # Attention softmax
        self.softmax = nn.Softmax(dim=-1)
        self.mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>")|(np.array(sents)[:, 1:] =='<s>')|(np.array(sents)[:, 1:] =='</s>')|(np.array(sents)[:, 1:] =='<mask>')|(np.array(sents)[:, 1:] =='unk')|(np.array(sents)[:, 1:] =='')|(np.array(sents)[:, 1:] =='Ċ')).type_as(image).bool()
        
    def forward(self, x):
        # Compute key, query, and value tensors
        key = self.key_conv(x)
        query = self.query_conv(x)
        value = self.value_conv(x)

        # Reshape keys, queries, and values
        # keys = keys.view(keys.size(0), keys.size(-1) * keys.size(-2) * keys.size(-3),-1)
        # queries = queries.view(queries.size(0), queries.size(-1) * queries.size(-2) * queries.size(-3),-1)
        # values = values.view(values.size(0), values.size(-1) * values.size(-2) * values.size(-3),-1)
        batch_size, channels, depth, height, width = x.size()
        query = query.view(batch_size, channels, -1)
        key = key.view(batch_size, channels, -1)
        value = value.view(batch_size, channels, -1)


        # Compute attention scores
        # print("qqq",queries.shape,keys.shape,values.shape)
        attention = torch.matmul(query.permute(0, 2, 1), key)  #keys.transpose(1, 2)
        attention = self.softmax(attention)

        # Apply attention to values
        # print("shapes",values.shape,attention.shape)
        # print("value",value.shape)
        out = torch.matmul(attention, value.permute(0, 2, 1))
        print("out_cros",out.shape)
        out = out.view(batch_size, channels, depth, height, width)
        out = out.permute(0,2,3,4,1)
        # Residual connection and scaling
        out = self.gamma * out + x
        out = out.view(batch_size, depth* height* width,channels)
        out=out.mean(1)
        # print("HII",out.shape,attention.shape)
        return out,attention



class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Define the key, query, and value linear transformations
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        # Attention softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Compute key, query, and value tensors
        key = self.key_conv(x)
        query = self.query_conv(x)
        value = self.value_conv(x)

        # Reshape keys, queries, and values
        # keys = keys.view(keys.size(0), keys.size(-1) * keys.size(-2) * keys.size(-3),-1)
        # queries = queries.view(queries.size(0), queries.size(-1) * queries.size(-2) * queries.size(-3),-1)
        # values = values.view(values.size(0), values.size(-1) * values.size(-2) * values.size(-3),-1)
        batch_size, channels, depth, height, width = x.size()
        query = query.view(batch_size, channels, -1)
        key = key.view(batch_size, channels, -1)
        value = value.view(batch_size, channels, -1)


        # Compute attention scores
        # print("qqq",queries.shape,keys.shape,values.shape)
        attention = torch.matmul(query.permute(0, 2, 1), key)  #keys.transpose(1, 2)
        attention = self.softmax(attention)

        # Apply attention to values
        # print("shapes",values.shape,attention.shape)
        # print("value",value.shape)
        out = torch.matmul(attention, value.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height, width)

        # Residual connection and scaling
        out = self.gamma * out + x
        out = out.view(batch_size, depth* height* width,channels)
        out=out.mean(1)
        # print("HII",out.shape,attention.shape)
        return out,attention

def propmt_generator(label):
    tokenizer = LongformerTokenizerFast.from_pretrained("yikuan8/Clinical-Longformer")
    actual_label= "braf fusion" if label == 1 else "braf v600e mutation"
    # text = f"this is a {actual_label} brain tumor genetic marker case"
    if label==1:
        # text = "This is BRAF fusion as it corresponds to a genetic anomaly where the BRAF gene fuses with another gene, resulting in an abnormal protein. This fusion may trigger uncontrolled cell growth, potentially leading to the development or progression of brain tumors."
        text = "BRAF fusions may involve at least nine different exon combinations, all of which contain an uncontrolled BRAF kinase domain resulting in a constitutively activated MAPK pathway. BRAF fused tumors, the activated RAF1 domain results in activation of the MAP kinase pathway. larger , and had a greater mass effect, increased frequency of hydrocephalus , and diffuse enhancement"
        # text = " Pediatric low-grade gliomas with BRAF fusion may exhibit features such as well-circumscribed tumor margins, heterogeneous enhancement, and mixed signal intensities on T1-weighted and T2-weighted images.Compared to low-grade gliomas without BRAF fusion, those with the fusion may present with less aggressive imaging features, including smaller size, more well-defined margins, and potentially slower growth rates.T1-weighted imaging can highlight the tumor's structure, while T2-weighted and FLAIR imaging can help assess edema and infiltration. Advanced modalities such as MR spectroscopy may reveal metabolic changes indicative of BRAF fusion.Tumor borders in pediatric low-grade glioma with BRAF fusion are often well-circumscribed and clearly defined, which may aid in surgical planning.Tumors with BRAF fusion may display slow growth rates and minimal invasive behavior on MR imaging, as opposed to more aggressive patterns in other pediatric brain tumors.MR imaging can show moderate enhancement with contrast in tumors with BRAF fusion, suggesting some degree of vascularity, although this may vary between cases. In response to treatment, changes in tumor size, enhancement, and edema can be observed on MR imaging. Tumors with BRAF fusion may show favorable responses to targeted therapies.Imaging features such as well-defined margins and smaller tumor sizes in pediatric low-grade gliomas with BRAF fusion may correlate with better prognosis and overall outcomes.Techniques like MR spectroscopy can provide metabolic insights, potentially showing altered metabolite levels associated with BRAF fusion in pediatric low-grade glioma.Long-term MR imaging follow-up may reveal stable tumor sizes and minimal changes in tumors with BRAF fusion, suggesting a favorable prognosis."
    else: 
        # text = "This is BRAF mutation as  it shows the creation of an abnormal protein that sends signals that lead to uncontrolled cell growth and cancer."
        text = "Braf v600e mutation because they acquired intrinsic epileptogenic properties in neuronal lineage cells during brain development, whereas tumorigenic properties were attributed to high proliferation of glial lineage cells and responded to BRAF V600E inhibitor therapy. A BRAF V600E mutation is the most common recurrent genetic alteration occurring in 20%–60% of all gangliogliomas. The BRAF V600E mutation results in an activated protein that signals to MEK–ERK constitutively, stimulating cell proliferation and survival. There is a strong association of V600E-mutant tumors with age. The V600E mutation frequency is higher in young patients and less common in anaplastic ganglioglioma variants. The BRAF V600E mutation is mainly seen in mutant ganglion cells but may also be present in the glial component, suggesting that both components are derived from a common precursor with early mutation acquisition.  hemispheric, appeared more infiltrative and, though infrequent, were the only group demonstrating diffusion restriction with a lower ADC ratio" #small molecule inhibitor treatment in metastasized melanoma"
        # text = "The BRAF V600E mutation in pediatric low-grade glioma is associated with certain MRI features, such as increased contrast enhancement and specific patterns of tumor growth. These features may help distinguish tumors with the BRAF V600E mutation from those without.  Tumors with the BRAF V600E mutation may show distinct MRI characteristics such as more pronounced contrast enhancement and a higher likelihood of cystic components. Non-mutated gliomas may present with more uniform or less aggressive imaging features.Advanced MRI techniques like perfusion-weighted imaging and MR spectroscopy may provide additional information about tumor vascularity and metabolic activity. These insights can help differentiate BRAF V600E-mutated gliomas from other types. MRI features such as tumor size, location, and enhancement patterns associated with the BRAF V600E mutation may correlate with clinical outcomes, including treatment response and prognosis. For example, increased enhancement may indicate a more aggressive tumor.A case study of a pediatric low-grade glioma with the BRAF V600E mutation may reveal MRI features such as irregular tumor borders, cystic areas, and variable contrast enhancement. Treatment approaches could include targeted therapies, and outcomes may vary based on the tumor's behavior and response to treatment.MRI can be used to monitor changes in tumor size, enhancement, and edema in response to treatment. Tumors with the BRAF V600E mutation may respond differently to targeted therapies, and MRI can help assess the effectiveness of these treatments. Potential MRI biomarkers for the BRAF V600E mutation in pediatric low-grade glioma could include patterns of contrast enhancement, perfusion characteristics, and metabolic profiles. These biomarkers could aid in non-invasive identification and guide treatment planning."
    # print("textttt",text)
    tokenized_text =tokenizer.encode_plus(text,return_tensors="pt",padding="max_length", max_length = 1159)
    return tokenized_text




def calc_similarity(global_text_embedding ,local_text_embedding, global_img_embedding, local_image_embedding, patch_atten_output , word_atten_output):

    global_img_embedding = global_img_embedding.detach().cpu().numpy()
    global_text_embedding = global_text_embedding.detach().cpu().numpy()
    bz = global_text_embedding.shape[0]
    global_similarity = metrics.pairwise.cosine_similarity(global_img_embedding, global_text_embedding)

    local_similarity_i_t = F.cosine_similarity(patch_atten_output.reshape(bz,-1).unsqueeze(1), local_image_embedding.reshape(bz,-1).unsqueeze(0), dim=-1)
    local_similarity_t_i = F.cosine_similarity(word_atten_output.reshape(bz,-1).unsqueeze(1), local_text_embedding.reshape(bz,-1).unsqueeze(0), dim=-1)

    local_similarity = local_similarity_i_t #+ local_similarity_t_i
    global_similarity = torch.Tensor(global_similarity).cuda()
    similarity = (global_similarity + local_similarity)/2
    return similarity




def zeroshot_classifier(label_list, image , model):
    
    # min_similarity = 0 
    model = model.cuda()
    # predicted_label = 0
    print("label_list",label_list)
    similarity_list = []
    for label in label_list:
        tokenized_label_text = propmt_generator(label)
        
        text_model = model.transformer
        img_model= model.cnn
        
        with torch.no_grad():
            image = image.float()
            img_feat_q, patch_feat_q  = img_model(image)
        
            local_image_embedding = img_model.local_embed(patch_feat_q) # ****
            local_image_embedding = F.normalize(local_image_embedding, dim=-1)   #****
                    
            global_img_embedding = img_feat_q
            global_img_embedding = F.normalize(global_img_embedding, dim=-1)




            attention_mask = tokenized_label_text['attention_mask']
            attention_mask = attention_mask.cuda()
            input_id = tokenized_label_text['input_ids'].squeeze(1)
            input_id = input_id.cuda()
            report_feat_q, word_feat_q, sents ,_ = text_model(input_id, attention_mask)
            print("word_feat_q",word_feat_q.shape)
            local_text_embedding = text_model.local_embed(word_feat_q)  
            local_text_embedding = F.normalize(local_text_embedding, dim=-1)
            global_text_embedding = text_model.global_embed(report_feat_q)
            global_text_embedding = F.normalize(global_text_embedding, dim=-1)
            padding_mask_list=['<s>', '</s>', '<pad>','<mask>' , 'unk','','Ċ']
            mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>")|(np.array(sents)[:, 1:] =='<s>')|(np.array(sents)[:, 1:] =='</s>')|(np.array(sents)[:, 1:] =='<mask>')|(np.array(sents)[:, 1:] =='unk')|(np.array(sents)[:, 1:] =='')|(np.array(sents)[:, 1:] =='Ċ')).type_as(image).bool()
            print("patchhh", local_image_embedding.shape , local_text_embedding.shape )
            patch_atten_output, patch_weights,_ = model.patch_local_atten_layer(local_image_embedding,local_text_embedding,sents)
            
            word_atten_output, word_weights,_ = model.word_local_atten_layer(local_image_embedding,local_text_embedding,sents)
            
            # patch_weights = F.normalize(patch_weights, dim=-1)
            # word_weights = F.normalize(word_weights, dim=-1)
            cap_lens = [
                len([w for w in sent if not w.startswith("[")]) -1 for sent in sents
            ] 
              
            
        similarity = calc_similarity(global_text_embedding ,local_text_embedding, global_img_embedding, local_image_embedding , patch_atten_output , word_atten_output)
        # print("simmm",similarity)
        # print("similarity",similarity)
        similarity_list.append(similarity)
        # if similarity>=min_similarity:
        #     predicted_label = label
    print("similarity_list",similarity_list)
    print("arayyy",np.array(similarity_list).shape)
    predicted_label = np.argmax(np.array(similarity_list))
    return predicted_label




def fewshot_classifier(query_image, support_dl , model):
    
    
    model = model.cuda()
   
    # similarity_list = []
    with torch.no_grad():
        img_model= model.cnn
        query_image = query_image.float()
        img_feat_q, patch_feat_q  = img_model(query_image)
    
        local_image_embedding = img_model.local_embed(patch_feat_q) # ****
        local_image_embedding = F.normalize(local_image_embedding, dim=-1)   #****
                
        global_img_embedding = img_feat_q
        global_img_embedding = F.normalize(global_img_embedding, dim=-1)
        max_sim = 0
        for support_images,labels in support_dl:
            
            support_images,labels = support_images.cuda().float(), labels.cuda().float()
            img_feat_q, patch_feat_q  = img_model(support_images)
    
            local_support_embedding = img_model.local_embed(patch_feat_q) # ****
            local_support_embedding = F.normalize(local_image_embedding, dim=-1)   #****
                    
            global_support_embedding = img_feat_q
            global_support_embedding = F.normalize(global_img_embedding, dim=-1)
            
        
        
       
            similarity = calc_similarity(global_support_embedding ,local_support_embedding, global_img_embedding, local_image_embedding , local_image_embedding , local_image_embedding)
            if similarity>max_sim:
                max_sim = similarity
                predicted_label = labels
                print("fewwwwwwwwwwwwwww",predicted_label,similarity)
            # similarity_list.append(similarity)
        # if similarity>=min_similarity:
        #     predicted_label = label
    # print("similarity_list",similarity_list)
    # print("arayyy",np.array(similarity_list).shape)
    # predicted_label = np.argmax(np.array(similarity_list))
    return predicted_label


             


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiHeadSelfAttention, self).__init__()

        num_heads = 2
        self.num_heads = num_heads
        head_dim = in_channels // num_heads

        # Define separate linear transformations for keys, queries, and values for each attention head
        self.key_conv = nn.ModuleList([nn.Conv3d(in_channels, head_dim, kernel_size=1) for _ in range(num_heads)])
        self.query_conv = nn.ModuleList([nn.Conv3d(in_channels, head_dim, kernel_size=1) for _ in range(num_heads)])
        self.value_conv = nn.ModuleList([nn.Conv3d(in_channels, head_dim, kernel_size=1) for _ in range(num_heads)])
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        batch_size, channels, height, width, depth = x.size()

        # Compute key, query, and value tensors for each attention head
        keys = [key_conv(x).view(batch_size, -1, depth) for key_conv in self.key_conv]
        queries = [query_conv(x).view(batch_size, -1, depth) for query_conv in self.query_conv]
        values = [value_conv(x).view(batch_size, -1, depth) for value_conv in self.value_conv]

        # Concatenate outputs of each attention head
        keys = torch.cat(keys, dim=1)
        queries = torch.cat(queries, dim=1)
        values = torch.cat(values, dim=1)

        # Compute attention scores
        attention = torch.matmul(queries.permute(0, 2, 1), keys)
        attention /= self.num_heads  # Scale by the number of heads
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # Apply attention to values
        out = torch.matmul(attention, values.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth)

        # Residual connection and scaling
        out = self.gamma * out + x

        return out, attention




class MultiHeadDepthAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiHeadDepthAttention, self).__init__()

        # Define the key, query, and value linear transformations

        num_heads = 2
        self.num_heads = num_heads
        # head_dim = in_channels // num_heads
        head_dim = in_channels*height*width // num_heads
        # Define separate linear transformations for keys, queries, and values for each attention head
        self.key_conv = nn.ModuleList([nn.Conv3d(in_channels, head_dim, kernel_size=1) for _ in range(num_heads)])
        self.query_conv = nn.ModuleList([nn.Conv3d(in_channels, head_dim, kernel_size=1) for _ in range(num_heads)])
        self.value_conv = nn.ModuleList([nn.Conv3d(in_channels, head_dim, kernel_size=1) for _ in range(num_heads)])
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.25)
    def forward(self, x):
        # Compute key, query, and value tensors
        # print("x_aten",x.shape)
        # Compute key, query, and value tensors for each attention head
        batch_size, channels,  height, width ,depth= x.size()
        
        keys = [key_conv(x).view(batch_size,-1, depth) for key_conv in self.key_conv]
        queries = [query_conv(x).view(batch_size, -1, depth) for query_conv in self.query_conv]
        values = [value_conv(x).view(batch_size, -1, depth) for value_conv in self.value_conv]

        keys = torch.cat(keys, dim=1)
        queries = torch.cat(queries, dim=1)
        values = torch.cat(values, dim=1)

        # Reshape keys, queries, and values
        # keys = keys.view(keys.size(0), keys.size(-1) * keys.size(-2) * keys.size(-3),-1)
        # queries = queries.view(queries.size(0), queries.size(-1) * queries.size(-2) * queries.size(-3),-1)
        # values = values.view(values.size(0), values.size(-1) * values.size(-2) * values.size(-3),-1)


       
        # x=x.mean(1)
        # batch_size, height, width, depth = x.size()
 

        # query = query.view(batch_size, channels, -1)
        # keys = key.view(batch_size, channels, -1)
        # value = value.view(batch_size, channels, -1)

        # query = query.view(batch_size, channels, -1)
        # key = key.view(batch_size, channels, -1)
        # value = value.view(batch_size, channels, -1)

        # query = query.view(batch_size, height*width*channels, -1)
        # # key = key.view(batch_size, height*width*channels, -1)

        # query = query.view(batch_size, channels*height*width,-1)
        # key = key.view(batch_size, channels*height*width,-1)
        # # query = query.permute(0,1,-1,2)
        # # key = key.permute(0,1,-1,2)

        # # query = query.mean(2)
        # # d_q = query.size(-1)
        # # key = key.mean(2)
        # # query = nn.AdaptiveAvgPool3d((4,512,1,5))(query).squeeze(3)
        # # key = nn.AdaptiveAvgPool3d((4,512,1,5))(key).squeeze(3)
        # # print("quey_key",query.shape,key.shape)

        # # query = F.avg_pool3d(query, kernel_size=2, stride=2, padding=0)
        # # key = F.max_pool3d(key, kernel_size=2, stride=2, padding=0)
        # # query = (nn.AdaptiveAvgPool3d((512,1,5))(query)+nn.AdaptiveMaxPool3d((512,1,5))(query)).squeeze(2)
        # # key = (nn.AdaptiveAvgPool3d((512,1,5))(key)+nn.AdaptiveMaxPool3d((512,1,5))(key)).squeeze(2)
        # # print("quey_key",query.shape,key.shape)
        # value = value.view(batch_size, height*width*channels, -1)
        # # pred_mask = torch.ones(batch_size, height*width*channels, depth).cuda()
        # # Compute attention scores
        # print("qqq",queries.shape,keys.shape,values.shape)
        # print("queryyyy",query.shape,key.shape)
        attention = torch.matmul(queries.permute(0, 2, 1), keys)
        attention /= self.num_heads  # Scale by the number of heads
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # Apply attention to values
        out = torch.matmul(attention, values.permute(0, 2, 1))
        out = out.view(batch_size, channels*height*width, depth)
        # atten_mask = torch.matmul(attention, pred_mask.permute(0, 2, 1))
        # atten_mask = atten_mask.view(batch_size, channels,  height, width,depth)
        # print("atten_mask",atten_mask.shape)
        # out = out.view(batch_size, height, width,depth)
        temp = out
        # Residual connection and scaling
        out = self.gamma * out + x
        # print("outtt",out.shape,attention.shape)

        # out = out.view(batch_size, depth* height* width,channels)

        # out = out.view(batch_size, height* width,depth,channels)
        # out=out.mean(1)
        # print("HII",out.shape,attention.shape)
        
        return out,attention#temp.mean(1)#temp.mean(1)#attention





# class CrossAttention(nn.Module):
#     def __init__(self, in_channels,mode,sents):
#         super(CrossAttention, self).__init__()

#         # Define the key, query, and value linear transformations
#         in_channels = 256
#         self.key_img= nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()
#         self.query_img = nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()
#         self.value_img = nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()


#         self.key_txt=  nn.Conv1d(in_channels, in_channels, kernel_size=1).cuda()
#         self.query_txt = nn.Conv1d(in_channels, in_channels, kernel_size=1).cuda()
#         self.value_txt =  nn.Conv1d(in_channels, in_channels, kernel_size=1).cuda()


#         # self.gamma = nn.Parameter(torch.randn(1)).cuda()  #torch.zeros(1)

#         # Attention softmax
#         # self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(p=0.25)
#         self.mode = mode
#         self.sents = sents
#         self.dropout = nn.Dropout(p=0.5)
#     def forward(self, img_embed, txt_embed):
#         # Compute key, query, and value tensors
#         # print("x_aten",x.shape)
#         # print("outtttttttttttttttt",txt_embed)
#         sents = self.sents
#         # self.mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>") |
#         #                      (np.array(sents)[:, 1:] == '<s>') |
#         #                      (np.array(sents)[:, 1:] == '</s>') |
#         #                      (np.array(sents)[:, 1:] == '<mask>') |
#         #                      (np.array(sents)[:, 1:] == 'unk') |
#         #                      (np.array(sents)[:, 1:] == '') |
#         #                      (np.array(sents)[:, 1:] == 'Ċ')).type(torch.bool)

# # Create a mask to filter tensor1
#         # mask = ~self.mask_pad.unsqueeze(-1)  # Expand mask to match tensor1's shape

# # Filter tensor1 based on the mask
        
#         # txt_embed = text_embed[mask]
        
#         self.mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>")|(np.array(sents)[:, 1:] =='<s>')|(np.array(sents)[:, 1:] =='</s>')|(np.array(sents)[:, 1:] =='<mask>')|(np.array(sents)[:, 1:] =='unk')|(np.array(sents)[:, 1:] =='')|(np.array(sents)[:, 1:] =='Ċ')).type_as(img_embed).bool()
#         # self.text_masks = torch.tensor([token in self.mask_pad for token in text_tokens], dtype=torch.bool)
#         # print("mask_pad",self.mask_pad)
#         # print("crossssssssss3",img_embed.shape, txt_embed.shape)
#         # print("txt_embed",txt_embed.shape)
#         if self.mode == "img_txt":
#             txt_embed = txt_embed.permute(0,-1,1)
#             # value = value.permute(0,-1,1)
#             key = self.key_txt(txt_embed)
#             # key = self.dropout(key)
#             query = self.query_img(img_embed)
#             # query = self.dropout(query)
#             value = self.value_txt(txt_embed)
#             # value = self.dropout(value)

#             # Reshape keys, queries, and values
#             # keys = keys.view(keys.size(0), keys.size(-1) * keys.size(-2) * keys.size(-3),-1)
#             # queries = queries.view(queries.size(0), queries.size(-1) * queries.size(-2) * queries.size(-3),-1)
#             # values = values.view(values.size(0), values.size(-1) * values.size(-2) * values.size(-3),-1)


#             batch_size, channels,  height, width ,depth= img_embed.size()
#             batch_size, text_len, embed_dim = txt_embed.size()
            
#             # query = query.permute(0,1,-1,2,3)
           
#             # value = value.view(batch_size, channels*depth, -1)
#     # ### this part
#             query = query.view(batch_size, channels, -1)#.mean(2)
#             # key = key.view(batch_size, channels*depth, -1)
#             # value = value.view(batch_size, depth*channels, -1)
#             attention = torch.bmm(query.permute(0, 2, 1), key)#/np.sqrt(d_q)  #keys.transpose(1, 2)
#             # attention[self.mask_pad.unsqueeze(1).expand(-1, attention.size(1),-1)] = float('-inf')
#             attention = attention.masked_fill(self.mask_pad.unsqueeze(1), -1e12)#float('-inf'))
#             # print("crossssssss4",attention.shape)
#             attention = torch.softmax(attention,dim = -1)
            
            
           
#             out = torch.bmm(value, attention.permute(0, 2, 1))
#             # attention = torch.bmm(query.permute(0, 2, 1), key)#/np.sqrt(d_q)  #keys.transpose(1, 2)
#             # # attention = nn.ReLU()(attention)
#             # attention = self.softmax(attention)
#             # print("outtttttttttttttttt",query)
            
#             # # Apply attention to values
#             # # print("shapes",values.shape,attention.shape)
#             # # print("value",value.shape)
#             # # print("value",value.shape,attention.shape)
#             # out = torch.matmul(attention, value.permute(0, 2, 1))
#             # print("crossssssssssssssssssss", out.shape)
#             out = out.view(batch_size, channels,  height, width,depth)

#             # atten_mask = mask_pred.view(batch_size, depth*channels, height*width)
#             # final_mask = torch.matmul(attention, atten_mask.permute(0, 2, 1))
#             # final_mask = final_mask.view(batch_size, channels,  height, width,depth)
#             # final_mask = out - value.view(batch_size, channels,  height, width,depth)
#             # print("finalllll", final_mask.shape)
#             # out = out.view(batch_size, height, width,depth)
#             temp = out
#             # Residual connection and scaling
#             # print("out111",torch.unique(self.gamma))



#             #####################333
#             # out = self.gamma * out + img_embed
#             ########################3


#             # print("outtt",out.shape,attention.shape)
#             # removed
#             # out = out.view(batch_size, depth* height* width,channels)
#             # print("att",out, attention, query)
#             return out,attention, query
#         elif self.mode == "txt_img":
#             txt_embed = txt_embed.permute(0,-1,1)
#             key = self.key_img(img_embed)
#             # key = self.dropout(key)
#             query = self.query_txt(txt_embed)
#             # query = self.dropout(query)
#             value = self.value_img(img_embed)
#             # value = self.dropout(value)

#             batch_size, channels,  height, width ,depth= img_embed.size()
            
#             # key = key.permute(0,1,-1,2,3)
#             # value = value.permute(0,1,-1,2,3)
#             # query = query.permute(0,-1,1)
#             value = value.view(batch_size, channels, -1)
  
#             # query = query.view(batch_size, channels*depth, -1)
#             key = key.view(batch_size, channels, -1)
#             attention = torch.bmm(query.permute(0, 2, 1), key)#/np.sqrt(d_q)  #keys.transpose(1, 2)
#             # attention = nn.ReLU()(attention)
#             # print("heyyy",attention.shape)
#             # print("maskkkkkkksksk",self.mask_pad.shape)
#             # print("!!!11", attention.shape,self.mask_pad.shape)
#             # .unsqueeze(1).expand(-1, -1, attention.size(-1))
#             # print("attention",attention)
#             # attention[self.mask_pad.unsqueeze(-1).expand(-1, -1, attention.size(-1))] = float('-inf')
#             attention = attention.masked_fill(self.mask_pad.unsqueeze(2),-1e12)#, float('-inf'))
#             attention = torch.softmax(attention,dim=-1)
            
#             # print("atttttttat",channels)
#             out = torch.bmm(value, attention.permute(0, 2, 1))
#             # print("outttttttt",out)
#             # attention = torch.matmul(query.permute(0, 2, 1), key)#/np.sqrt(d_q)  #keys.transpose(1, 2)
#             # # attention = nn.ReLU()(attention)
#             # attention = self.softmax(attention)


            
#             # out = torch.matmul(attention, value.permute(0, 2, 1))
#             # print("crosssss2",out.shape)
#             out = out.view(batch_size, channels,  1158)

#             temp = out
#             # Residual connection and scaling
#             # print("out2222",torch.unique(self.gamma))


#             # out = self.gamma * out + txt_embed#query


#             # print("att",out.permute(0,-1,1), attention, query.permute(0,-1,1))
#             return out.permute(0,-1,1), attention, query.permute(0,-1,1)
        







class CrossAttention(nn.Module):
    def __init__(self, in_channels,mode):#,sents):
        super(CrossAttention, self).__init__()

        # Define the key, query, and value linear transformations
        in_channels = 256
        self.key_img= nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()
        self.query_img = nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()
        self.value_img = nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()

        in_channels_text = 768
        self.key_txt=  nn.Conv1d(in_channels_text, in_channels, kernel_size=1).cuda()
        self.query_txt = nn.Conv1d(in_channels_text, in_channels, kernel_size=1).cuda()
        self.value_txt =  nn.Conv1d(in_channels_text, in_channels, kernel_size=1).cuda()


        self.gamma = nn.Parameter(torch.randn(1)).cuda()  #torch.zeros(1)
        # self.gamma2 = nn.Parameter(torch.randn(1)).cuda() 
        # Attention softmax
        # self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.25)
        self.mode = mode
        # self.sents = sents
        self.dropout = nn.Dropout(p=0.5)
        self.d_out_kq = 256
    def forward(self, img_embed, txt_embed,sents):
        # Compute key, query, and value tensors
        # print("x_aten",img_embed.shape,txt_embed.shape)
        # print("outtttttttttttttttt",txt_embed)


        # sents = self.sents


        # self.mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>") |
        #                      (np.array(sents)[:, 1:] == '<s>') |
        #                      (np.array(sents)[:, 1:] == '</s>') |
        #                      (np.array(sents)[:, 1:] == '<mask>') |
        #                      (np.array(sents)[:, 1:] == 'unk') |
        #                      (np.array(sents)[:, 1:] == '') |
        #                      (np.array(sents)[:, 1:] == 'Ċ')).type(torch.bool)

# Create a mask to filter tensor1
        # mask = ~self.mask_pad.unsqueeze(-1)  # Expand mask to match tensor1's shape

# Filter tensor1 based on the mask
        
        # txt_embed = text_embed[mask]
        
        self.mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>")|(np.array(sents)[:, 1:] =='<s>')|(np.array(sents)[:, 1:] =='</s>')|(np.array(sents)[:, 1:] =='<mask>')|(np.array(sents)[:, 1:] =='unk')|(np.array(sents)[:, 1:] =='')|(np.array(sents)[:, 1:] =='Ċ')).type_as(img_embed).bool()
        # self.text_masks = torch.tensor([token in self.mask_pad for token in text_tokens], dtype=torch.bool)
        # print("mask_pad",self.mask_pad)
        # print("crossssssssss3",img_embed.shape, txt_embed.shape)
        # print("txt_embed",txt_embed.shape)
        if self.mode == "img_txt":
            # print("txrtttttttttttttttt1",txt_embed.shape)
            txt_embed = txt_embed.permute(0,-1,1)
            # value = value.permute(0,-1,1)
            # key =txt_embed# self.key_txt(txt_embed)
            key = self.key_txt(txt_embed)
            # key = self.dropout(key)
            # query = img_embed#self.query_img(img_embed)
            query = self.query_img(img_embed)
            # query = self.dropout(query)
            # value = txt_embed#self.value_txt(txt_embed)
            value = self.value_txt(txt_embed)
            # value = self.dropout(value)

            # Reshape keys, queries, and values
            # keys = keys.view(keys.size(0), keys.size(-1) * keys.size(-2) * keys.size(-3),-1)
            # queries = queries.view(queries.size(0), queries.size(-1) * queries.size(-2) * queries.size(-3),-1)
            # values = values.view(values.size(0), values.size(-1) * values.size(-2) * values.size(-3),-1)
            # print("newwwwwwwwwwwwwwwwww",img_embed.shape,txt_embed.shape)

            batch_size, channels,  height, width ,depth= img_embed.size()
            # batch_size, text_len, embed_dim = txt_embed.size()
            batch_size, embed_dim,text_len = txt_embed.size()
            
            # query = query.permute(0,1,-1,2,3)
           
            # value = value.view(batch_size, channels*depth, -1)
    # ### this part
            query = query.view(batch_size, channels, -1)#.mean(2)
            # key = key.view(batch_size, channels*depth, -1)
            # value = value.view(batch_size, depth*channels, -1)



            attention = torch.bmm(query.permute(0, 2, 1), key)#/np.sqrt(d_q)  #keys.transpose(1, 2)
            # print("thereeeeeeeeeee", key.shape,query.shape)
            # attention = torch.bmm(query, key.permute(0, 2, 1))


            # attention[self.mask_pad.unsqueeze(1).expand(-1, attention.size(1),-1)] = float('-inf')
            # print("attt",attention.shape)
            # print("maskkkkkkkkkk2",self.mask_pad.shape)

            attention = attention.masked_fill(self.mask_pad.unsqueeze(1), -1e12)#float('-inf'))
            # print("ayyyyyyyyyy",attention[:,200,:])
            # print("crossssssss4",attention.shape,value.shape)
            attention = torch.softmax(attention,dim = -1)
            
            #/self.d_out_kq**0.5
           
            out = torch.bmm(value, attention.permute(0, 2, 1))
            # out = torch.bmm(attention, value)

            # attention = torch.bmm(query.permute(0, 2, 1), key)#/np.sqrt(d_q)  #keys.transpose(1, 2)
            # # attention = nn.ReLU()(attention)
            # attention = self.softmax(attention)
            # print("outtttttttttttttttt",query)
            
            # # Apply attention to values
            # print("shapes",out.shape,attention.shape)
            # # print("value",value.shape)
            # # print("value",value.shape,attention.shape)
            # out = torch.matmul(attention, value.permute(0, 2, 1))
            # print("crossssssssssssssssssss", out.shape)
            out = out.view(batch_size, channels,  height, width,depth)

            # atten_mask = mask_pred.view(batch_size, depth*channels, height*width)
            # final_mask = torch.matmul(attention, atten_mask.permute(0, 2, 1))
            # final_mask = final_mask.view(batch_size, channels,  height, width,depth)
            # final_mask = out - value.view(batch_size, channels,  height, width,depth)
            # print("finalllll", final_mask.shape)
            # out = out.view(batch_size, height, width,depth)
            temp = out
            # Residual connection and scaling
            # print("out111",torch.unique(self.gamma))



            #####################333
            out = self.gamma * out + img_embed
            ########################3
            # print("gammmma",self.gamma)

            # print("outtt",out.shape,attention.shape)
            # removed
            # out = out.view(batch_size, depth* height* width,channels)
            # print("att",out, attention, query)
            # print("gammmmaaa1",temp.shape)
            return out,attention, query#img_embed#
        elif self.mode == "txt_img":
            # print("shapee4",txt_embed.shape , img_embed.shape)
            txt_embed = txt_embed.permute(0,-1,1)
            # key = img_embed#self.key_img(img_embed)
            key = self.key_img(img_embed)
            # key = self.dropout(key)
            # query = txt_embed#self.query_txt(txt_embed)

            query = self.query_txt(txt_embed)
            # query = self.dropout(query)
            # value = img_embed#self.value_img(img_embed)
            value = self.value_img(img_embed)
            # value = self.dropout(value)

            batch_size, channels,  height, width ,depth= img_embed.size()
            
            # key = key.permute(0,1,-1,2,3)
            # value = value.permute(0,1,-1,2,3)
            # query = query.permute(0,-1,1)
            value = value.view(batch_size, channels, -1)
  
            # query = query.view(batch_size, channels*depth, -1)
            key = key.view(batch_size, channels, -1)
            attention = torch.bmm(query.permute(0, 2, 1), key)#/np.sqrt(d_q)  #keys.transpose(1, 2)
            # attention = torch.bmm(query, key.permute(0, 2, 1))

            # attention = nn.ReLU()(attention)
            # print("heyyy",attention.shape)
            # print("maskkkkkkksksk",self.mask_pad.shape)
            # print("!!!11", attention.shape,self.mask_pad.shape)
            # .unsqueeze(1).expand(-1, -1, attention.size(-1))
            # print("attention",attention)
            # attention[self.mask_pad.unsqueeze(-1).expand(-1, -1, attention.size(-1))] = float('-inf')
            # print("attt2",attention.shape)
            # print("maskkkkkkkkkk2222222222222",self.mask_pad.shape)
            attention = attention.masked_fill(self.mask_pad.unsqueeze(2),-1e12)#, float('-inf'))
            attention = torch.softmax(attention,dim=-1)
            
            # print("atttttttat",channels)
            out = torch.bmm(value, attention.permute(0, 2, 1))
            # out = torch.bmm(value, attention)

            # print("outttttttt",out.shape)
            # attention = torch.matmul(query.permute(0, 2, 1), key)#/np.sqrt(d_q)  #keys.transpose(1, 2)
            # # attention = nn.ReLU()(attention)
            # attention = self.softmax(attention)


            
            # out = torch.matmul(attention, value.permute(0, 2, 1))
            # print("crosssss2",out.shape)
            # print("outtttttttttttttttt2",out.shape,query.shape)
            out = out.view(batch_size, channels,  1158) #1158 # 1046

            temp = out
            # Residual connection and scaling
            # print("out2222",torch.unique(self.gamma))

                ############################################################################
            out = self.gamma * out + query#txt_embed#query

            # print("gammmmaaa2",self.gamma)
            # print("gammmmaaa2",temp.shape)
            # print("att",out.permute(0,-1,1), attention, query.permute(0,-1,1))  #txt_embed
            return out.permute(0,-1,1), attention, query.permute(0,-1,1)


class CrossAttention2(nn.Module):
    def __init__(self, in_channels,mode):#,sents):
        super(CrossAttention2, self).__init__()

        # Define the key, query, and value linear transformations
        in_channels = 256
        # self.key_img= nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()
        # self.query_img = nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()
        # self.value_img = nn.Conv3d(in_channels, in_channels, kernel_size=1).cuda()


        # self.key_txt=  nn.Conv1d(in_channels, in_channels, kernel_size=1).cuda()
        # self.query_txt = nn.Conv1d(in_channels, in_channels, kernel_size=1).cuda()
        # self.value_txt =  nn.Conv1d(in_channels, in_channels, kernel_size=1).cuda()

        self.temp1=4
        # self.gamma = nn.Parameter(torch.randn(1)).cuda()  #torch.zeros(1)
        # self.gamma2 = nn.Parameter(torch.randn(1)).cuda() 
        # Attention softmax
        # self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.25)
        self.mode = mode
        # self.sents = sents
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, img_embed, txt_embed):
        # Compute key, query, and value tensors
        # print("x_aten",img_embed.shape,txt_embed.shape)
        # print("outtttttttttttttttt",txt_embed)
        # sents = self.sents
        # self.mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>") |
        #                      (np.array(sents)[:, 1:] == '<s>') |
        #                      (np.array(sents)[:, 1:] == '</s>') |
        #                      (np.array(sents)[:, 1:] == '<mask>') |
        #                      (np.array(sents)[:, 1:] == 'unk') |
        #                      (np.array(sents)[:, 1:] == '') |
        #                      (np.array(sents)[:, 1:] == 'Ċ')).type(torch.bool)

# Create a mask to filter tensor1
        # mask = ~self.mask_pad.unsqueeze(-1)  # Expand mask to match tensor1's shape

# Filter tensor1 based on the mask
        
        # txt_embed = text_embed[mask]
        
        # self.mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>")|(np.array(sents)[:, 1:] =='<s>')|(np.array(sents)[:, 1:] =='</s>')|(np.array(sents)[:, 1:] =='<mask>')|(np.array(sents)[:, 1:] =='unk')|(np.array(sents)[:, 1:] =='')|(np.array(sents)[:, 1:] =='Ċ')).type_as(img_embed).bool()
        # self.text_masks = torch.tensor([token in self.mask_pad for token in text_tokens], dtype=torch.bool)
        # print("mask_pad",self.mask_pad)
        # print("crossssssssss3",img_embed.shape, txt_embed.shape)
        # print("txt_embed",txt_embed.shape)
        if self.mode == "img_txt":
            txt_embed = txt_embed.permute(0,-1,1)
            # value = value.permute(0,-1,1)
            key = self.key_txt(txt_embed)
            # key = self.dropout(key)
            query = self.query_img(img_embed)
            # query = self.dropout(query)
            value = self.value_txt(txt_embed)
            # value = self.dropout(value)

            # Reshape keys, queries, and values
            # keys = keys.view(keys.size(0), keys.size(-1) * keys.size(-2) * keys.size(-3),-1)
            # queries = queries.view(queries.size(0), queries.size(-1) * queries.size(-2) * queries.size(-3),-1)
            # values = values.view(values.size(0), values.size(-1) * values.size(-2) * values.size(-3),-1)


            batch_size, sourceL = key.size(0), key.size(2)
            ih, iw,id = query.size(2), query.size(3),query.size(4)
            queryL = ih * iw * id

            # --> batch x sourceL x ndf
            key = key.view(batch_size, -1, sourceL)
            query = query.view(batch_size, -1, queryL)
            contextT = torch.transpose(key, 1, 2).contiguous()

            # Get attention
            # (batch x sourceL x ndf)(batch x ndf x queryL)
            # -->batch x sourceL x queryL
            attn = torch.bmm(contextT, query)
            # --> batch*sourceL x queryL
            
            attn = attn.view(batch_size * sourceL, queryL)

            attn = nn.Softmax(dim=-1)(attn)

            # --> batch x sourceL x queryL
            
            attn = attn.view(batch_size, sourceL, queryL)
            # --> batch*queryL x sourceL
            attn = torch.transpose(attn, 1, 2).contiguous()
            attn = attn.view(batch_size * queryL, sourceL)

            attn = attn * self.temp1
            
            attn = nn.Softmax(dim=-1)(attn)
            attn = attn.view(batch_size, queryL, sourceL)
            # --> batch x sourceL x queryL
            attnT = torch.transpose(attn, 1, 2).contiguous()

            # (batch x ndf x sourceL)(batch x sourceL x queryL)
            # --> batch x ndf x queryL
            weightedContext = torch.bmm(key, attnT)
            # print("attn",weightedContext.shape)
            return weightedContext, attn.view(batch_size, -1, ih, iw,id),query
        
        elif self.mode == "txt_img":
            # print("texttttttttttttt",txt_embed.shape)
            txt_embed = txt_embed.permute(0,-1,1)
            key = img_embed#self.key_img(img_embed)
            # key = self.dropout(key)
            query = txt_embed#self.query_txt(txt_embed)
            # query = self.dropout(query)
            value = img_embed#self.value_img(img_embed)
            # value = self.dropout(value)

            batch_size, queryL = query.size(0), query.size(2)
            ih, iw,id = key.size(2), key.size(3),key.size(4)
            sourceL = ih * iw * id

            # --> batch x sourceL x ndf
            key = key.view(batch_size, -1, sourceL)
            query = query.view(batch_size, -1, queryL)
            contextT = torch.transpose(key, 1, 2).contiguous()

            # Get attention
            # (batch x sourceL x ndf)(batch x ndf x queryL)
            # -->batch x sourceL x queryL
            attn = torch.bmm(contextT, query)
            # --> batch*sourceL x queryL
            
            attn = attn.view(batch_size * sourceL, queryL)

            attn = nn.Softmax(dim=-1)(attn)

            # --> batch x sourceL x queryL
            
            attn = attn.view(batch_size, sourceL, queryL)
            # --> batch*queryL x sourceL
            attn = torch.transpose(attn, 1, 2).contiguous()
            attn = attn.view(batch_size * queryL, sourceL)

            attn = attn * self.temp1
            
            attn = nn.Softmax(dim=-1)(attn)
            attn = attn.view(batch_size, queryL, sourceL)
            # --> batch x sourceL x queryL
            attnT = torch.transpose(attn, 1, 2).contiguous()

            # (batch x ndf x sourceL)(batch x sourceL x queryL)
            # --> batch x ndf x queryL
            weightedContext = torch.bmm(key, attnT)
            # print("attn",weightedContext.shape)
            return weightedContext, attn.view(batch_size, -1, ih, iw,id),txt_embed#query
