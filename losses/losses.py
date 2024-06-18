
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pytorch_metric_learning import losses
from pytorch_metric_learning.losses import SelfSupervisedLoss,NTXentLoss
from pytorch_metric_learning.distances import CosineSimilarity
from fused_model import CrossAttention





# Source: https://github.com/mshaikh2/JoImTeR_MLMI_2021/tree/main
def cosine_distance(x1, x2, dim=1, eps=1e-8):
    """Returns (1 - cosine similarity) between x1 and x2, computed along dim.
    """
    # add a separate dimension to the tensors
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    cosine_similarity=w12 / (w1 * w2)
    cosine_distance = 1 - (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    return cosine_distance, cosine_similarity
# Source: https://github.com/mshaikh2/JoImTeR_MLMI_2021/tree/main
def triplet_loss_with_cosine_distance(anc, pos, neg, loc_label, margin,mode):
    positive_dist,_ = cosine_distance(anc, pos)
    negative_dist,_ =  cosine_distance(anc, neg)
    # cosine_loss = positive_dist - negative_dist + margin
    if mode=="global":
        cosine_loss = positive_dist - (1-loc_label)*negative_dist - 0.5*(loc_label)*negative_dist + margin
        
    elif mode=="local":
        cosine_loss = positive_dist - negative_dist  + margin
    #     z = torch.zeros_like(score)
    return nn.ReLU()(cosine_loss)






class ContrastiveLoss_cosine2(nn.Module):
    def __init__(self, margin,mode):
        super(ContrastiveLoss_cosine2, self).__init__()
        self.margin = margin
        self.mode=mode
#### power of 2 choices, vectorized
    def hard_negative_sampler(self,anchor,embedding,ids):
        hard_negative_list = []
        for i in range(len(ids)):
            min_distance = 1e12
            random_ids = np.random.choice(ids[ids!=i] ,size=2, replace=False)
            # print("random_ids",random_ids)
            for j in random_ids:#range(len(random_ids)):
                # if j!=i:
                    # print("anchorrr",anchor.shape,embedding.shape)
                # print("j",j)
                distance , _ = cosine_distance(anchor[i],embedding[j],dim=0) 
                
                if distance <= min_distance:
                    
                    min_distance = distance
                    hard_negative_id = j
            hard_negative_list.append(hard_negative_id)
        print("hard negatuve_list",hard_negative_list)
        return hard_negative_list

    def forward(self, img_embedding, text_embedding, labels,locations):#(self,cosine_similarity,labels):#
        # cosine_similarity = F.cosine_similarity(embedding1, embedding2)
        # print("heyyy",text_embedding.shape,img_embedding.shape)
        
        # Source: https://github.com/mshaikh2/JoImTeR_MLMI_2021/tree/main
        batch_size = img_embedding.shape[0]
        ids = np.array(list(range(batch_size)))
        eps=1e-8
        temp3=0.5#0.5#10.0
        # embedding1 = embedding1.unsqueeze(0)
        labels=Variable(torch.LongTensor(range(batch_size)))
        anchor_ids = labels
        positive_ids = labels
        img_anchor = img_embedding[anchor_ids]
        text_anchor = text_embedding[anchor_ids]
        # if batch_size!=1:
        # negative_ids = []
        
        
        # text_negative_ids = Variable(torch.LongTensor([torch.where(argmin_index < j, argmin_index, argmin_index + 1)argmin(cosine_distance(img_anchor,text_embedding[j for j in range(len(ids)) and j!=x ]))  for x in ids])) #neg_ids # hard negative sampling
        # text_negative_ids = Variable(torch.LongTensor([torch.argsort(cosine_distance(img_anchor,text_embedding[j for j in range(len(ids)) and j!=x ]))  for x in ids])) #neg_ids # hard negative sampling

        # image_negative_ids = Variable(torch.LongTensor([cosine_distance(for j!=x for x in ids])) #neg_ids # hard negative sampling
        
        image_negative_ids = Variable(torch.LongTensor(self.hard_negative_sampler(text_anchor,img_embedding,ids)))
        text_negative_ids = Variable(torch.LongTensor(self.hard_negative_sampler(img_anchor,text_embedding,ids)))
        # negative_ids = Variable(torch.LongTensor([np.random.choice(ids[ids!=x] ,size=2, replace=False) for x in ids]) #neg_ids
        text_negative = text_embedding[text_negative_ids]
        img_negative = img_embedding[image_negative_ids]
        location_negative_image = locations[text_negative_ids]
        location_negative_text = locations[image_negative_ids]
        
        i2t_triplet_loss = triplet_loss_with_cosine_distance(img_anchor, text_positive, text_negative,weights_img,
                                                            margin=self.margin,mode=self.mode)#self.margin)
        t2i_triplet_loss = triplet_loss_with_cosine_distance(text_anchor, img_positive, img_negative,weights_text,
                                                            margin=self.margin,mode=self.mode)#self.margin)
       
        i2t_cosine , t2i_cosine = cosine_distance(img_embedding,text_embedding)[1] , cosine_distance(text_embedding,img_embedding)[1]
        return i2t_triplet_loss.mean()+t2i_triplet_loss.mean(),i2t_cosine,t2i_cosine
    

        













def cosine_similarity(embedding1, embedding2, dim=1, eps=1e-8):  # extracted from: https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/gloria/loss/gloria_loss.py#L8  
    #forglobal_local
    """Returns cosine similarity between x1 and x2, computed along dim."""
    # print("embedding 1 and 2",embedding1.shape,embedding2.shape)
    embedding1=embedding1.unsqueeze(0)
    embedding2=embedding2.unsqueeze(0)
    embedding1_norm = torch.norm(embedding1, 2, dim=2, keepdim=True)
    embedding2_norm = torch.norm(embedding2, 2, dim=2, keepdim=True)

    cosine_similarity = torch.bmm(embedding1, embedding2.transpose(1, 2))
    norm0 = torch.bmm(embedding1_norm, embedding2_norm.transpose(1, 2))
    cosine_similarity = cosine_similarity / norm0.clamp(min=eps) 
    cosine_similarity = cosine_similarity.squeeze()
        
    return cosine_similarity



def cosine_similarity2(x1, x2, dim=1, eps=1e-8): # # extracted from: https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/gloria/loss/gloria_loss.py#L8  
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def local_contrastive_loss2(
    img_features, word_embedds,weiContext, cap_lens,locations,margin,temp1=4.0, temp2=5.0, temp3=0.5, agg="sum"
):
    # parts of this function was extracted from: https://github.com/marshuang80/gloria/blob/main/gloria/loss/gloria_loss.py 
    # print("img_fe",img_features.shape,word_embedds.shape,weiContext.shape)
    batch_size = img_features.shape[0]
    
    # att_maps = []
    similarities = []
    
    # for i in range(words_emb.shape[0]):

        # Get the i-th text description
    words_num = word_embedds.shape[1]
    
    # word = word_embedds[:, :, :words_num]#.unsqueeze(0).contiguous()  # [1, 768, 949]
        # word = word.repeat(batch_size, 1, 1)  # [4, 768, 949]
    context = img_features  # [48, 768, 19, 19]

        # weiContext, attn = attention_fn(
        #     word, context, temp1
        # )  # [48, 768, 25], [48, 25, 19, 19] 

        # att_maps.append(
        #     attn[i].unsqueeze(0).contiguous()
        # )  # add attention for curr index  [25, 19, 19]
    
    # word = word_embedds.transpose(1, 2).contiguous()  # [48, 25, 768]
    # weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]
    # print("kk",word.shape)
    # word = word.view(batch_size * words_num, -1)  # [1200, 768]
    # weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
    # print("now word",word.shape,weiContext.shape)
    
    word_embedds = word_embedds.view(batch_size, -1)  # [1200, 768]
    weiContext = weiContext.view(batch_size, -1)  # [1200, 768]
    #from here
    # row_sim = cosine_similarity(word_embedds, weiContext)
    # # print("simsim",row_sim.shape)
    # # # print("now2",row_sim.shape)
    # # row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

    # # row_sim.mul_(temp2).exp_()
    # # if agg == "sum":
    # #     row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
    # # else:
    # #     row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
    # # row_sim = torch.log(row_sim)

    #     # similarities.append(row_sim)
    # # print("similarity",row_sim.shape)
    # similarities = row_sim#torch.cat(similarities, 1)  #
    # similarities = similarities * temp3
    # labs = torch.arange(batch_size).long() #global
    # labels = torch.eye(batch_size)[labs].cuda() #global
    # # loss0 = torch.mean((1 - labels) * torch.pow(1 - similarities, 2) +
    # #                           (labels) * torch.pow(torch.clamp(similarities - 0.1, min=0.0), 2))
    # loss0 = torch.mean((1 - labels) * torch.pow(similarities, 2) +(labels) * torch.pow(torch.clamp(-similarities + 0.1, min=0.0), 2))
    # # similarities1 = similarities.transpose(0, 1)  # [48, 48]

    # # labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

    # # loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    # # loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    # labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)
    # if batch_size==1:
    #     # similarities=similarities.unsqueeze(0)
    #     loss_contrastive=loss0
    #     print("P")
    # else:
            
    #     loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    #to here
    labs = torch.arange(batch_size).long() #global
    labels = torch.eye(batch_size)[labs].cuda() #global
    c_loss = ContrastiveLoss_cosine2(margin=0.4,mode="local")
    # c_loss=local_entropy()
    # loss0,_,_=c_loss(word_embedds, weiContext,labels)
    # print("hereeetheree",word_embedds.shape, weiContext.shape)
    # print("hereeee_losss", weiContext,word_embedds)
    
    loss0,_,_=c_loss(word_embedds, weiContext,labels,locations)
    # print("wwwwwwwww",w_at.view(-1).shape,c_loss(word_embedds, weiContext,labels).shape)
    # print(w_at)
    # loss0=torch.sum(c_loss(word_embedds, weiContext,labels)[0]* w_at.view(-1))/ batch_size
    # loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return loss0#+loss1#, loss1, att_maps 




















def local_loss(
    img_features, words_emb, cap_lens,locations, margin=0.25,temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):

# parts of this function was extracted from: https://github.com/marshuang80/gloria/blob/main/gloria/loss/gloria_loss.py 
    batch_size = img_features.shape[0]

    att_maps = []
    wei_list = []
    words = []
   
    for i in range(words_emb.shape[0]):

    
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
        word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
        context = img_features  # [48, 768, 19, 19]

        
        crossAttention = CrossAttention(512,"txt_img")#,sents)
        weiContext, attn,_ = crossAttention(
            context,word)
          
        

    
    wei_list = torch.cat(wei_list, dim=0) 
    words = torch.cat(words, dim=0) 
    wei_list=wei_list.reshape(batch_size,-1)
    words=words.reshape(batch_size,-1)
    
   
    labs = torch.arange(batch_size).long() #global
    labels = torch.eye(batch_size)[labs].cuda() #global
    c_loss = ContrastiveLoss_cosine2(margin=margin,mode="local")
    loss0,_,_=c_loss(words, wei_list,labels,locations)
    return loss0#, loss1, att_maps
