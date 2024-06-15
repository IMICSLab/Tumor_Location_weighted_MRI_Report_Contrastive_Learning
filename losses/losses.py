
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pytorch_metric_learning import losses
from pytorch_metric_learning.losses import SelfSupervisedLoss,NTXentLoss
from pytorch_metric_learning.distances import CosineSimilarity
from fused_model import CrossAttention



# sbatch -N 1 -c 2 --mem 128G --time 72:00:00 --gres gpu:1 --array 1-1    --tmp=300G script_slurm.sh


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

def triplet_loss_with_cosine_distance(anc, pos, neg, loc_label, margin,mode):
    positive_dist,_ = cosine_distance(anc, pos)
    negative_dist,_ =  cosine_distance(anc, neg)
    # cosine_loss = positive_dist - negative_dist + margin
    if mode=="global":
        # cosine_loss = positive_dist - (1-loc_label)*negative_dist - 0.5*(loc_label)*negative_dist + margin
        cosine_loss = positive_dist - negative_dist  + margin
    elif mode=="local":
        cosine_loss = positive_dist - negative_dist  + margin
    #     z = torch.zeros_like(score)
    return nn.ReLU()(cosine_loss)




####https://github.com/adambielski/siamese-triplet/blob/master/utils.py:
# class HardNegativePairSelector():
#     """
#     Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
#     matching the number of positive pairs.
#     """

#     def __init__(self, cpu=True):
        

#     def get_pairs(self, img_embedding, text_embedding,img_anchor, text_anchor, labels):
        
#         min_dictance = 1e12
#         distance = cosine_distance( img_embedding, text_embedding)
#         for i in labels:
#             if i!=

#         # # labels = labels.cpu().data.numpy()
#         # all_pairs = np.array(list(combinations(range(len(labels)), 2)))
#         # all_pairs = torch.LongTensor(all_pairs)
#         # positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
#         negative_texts = text_embedding[i for i in labels ]

#         negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
#         negative_distances = negative_distances.cpu().data.numpy()
#         top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
#         top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

#         return top_negative_text, top_negative_image



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
        # else:
        #     text_negative = torch.zeros(batch_size).unsqueeze(0).cuda()
            
        #     img_negative = torch.zeros(batch_size).unsqueeze(0).cuda()
        #     location_negative = torch.zeros(batch_size).unsqueeze(0).cuda()
        # print("anchor_iddsssss",anchor_ids,negative_ids,positive_ids)
        # img_anchor = img_embedding[anchor_ids]
        text_positive = text_embedding[positive_ids]
        location_anchor = locations[anchor_ids]
        # location_positive = locations[positive_ids]
        

        # text_anchor = text_embedding[anchor_ids]
        img_positive = img_embedding[positive_ids]
        # print("longgggggg",negative_ids.shape)
        # print("location an", location_anchor,location_negative)
        weights_img = torch.tensor([1 if location_anchor[i] == location_negative_image[i] else 0 for i in range(len(location_anchor))]).cuda()
        weights_text = torch.tensor([1 if location_anchor[i] == location_negative_text[i] else 0 for i in range(len(location_anchor))]).cuda()
        # weights = [1  for i in range(len(location_anchor))]
        # margin_image = [0.2 if location_anchor[i] == location_negative_image[i] else 0.4 for i in range(len(location_anchor))]
        # margin_image = torch.Tensor(margin_image)
        # margin_image = margin_image.cuda()
        # margin_text = [0.2 if location_anchor[i] == location_negative_text[i] else 0.4 for i in range(len(location_anchor))]
        # margin_text = torch.Tensor(margin_text)
        # margin_text = margin_text.cuda()
        # i2t_triplet_loss = weights*triplet_loss_with_cosine_distance(img_anchor, text_positive, text_negative,
        #                                                     margin=self.margin)
        # t2i_triplet_loss = weights*triplet_loss_with_cosine_distance(text_anchor, img_positive, img_negative,
        #                                                     margin=self.margin)

        ######$$$$$$$
        i2t_triplet_loss = triplet_loss_with_cosine_distance(img_anchor, text_positive, text_negative,weights_img,
                                                            margin=self.margin,mode=self.mode)#self.margin)
        t2i_triplet_loss = triplet_loss_with_cosine_distance(text_anchor, img_positive, img_negative,weights_text,
                                                            margin=self.margin,mode=self.mode)#self.margin)
        ###########$$$$$$$$$$$$$$$4444
        # else: 
        #     i2t_triplet_loss = triplet_loss_with_cosine_distance(img_anchor, text_positive, text_negative,
        #                                                         margin=self.margin)/2
        #     t2i_triplet_loss = triplet_loss_with_cosine_distance(text_anchor, img_positive, img_negative,
        #                                                         margin=self.margin)/2
        i2t_cosine , t2i_cosine = cosine_distance(img_embedding,text_embedding)[1] , cosine_distance(text_embedding,img_embedding)[1]
        return i2t_triplet_loss.mean()+t2i_triplet_loss.mean(),i2t_cosine,t2i_cosine
    
    # def forward(self, img_embedding, text_embedding, labels,locations):#(self,cosine_similarity,labels):#
    #     # cosine_similarity = F.cosine_similarity(embedding1, embedding2)
    #     # print("heyyy")
        
    #     batch_size = img_embedding.shape[0]
    #     ids = np.array(list(range(batch_size)))
    #     eps=1e-8
    #     temp3=0.5#0.5#10.0
    #     # embedding1 = embedding1.unsqueeze(0)
    #     labels=Variable(torch.LongTensor(range(batch_size)))
    #     anchor_ids = labels
    #     positive_ids = labels
    #     img_anchor = img_embedding[anchor_ids]
    #     text_anchor = text_embedding[anchor_ids]
    #     # if batch_size!=1:
    #     # negative_ids = []
        
        
    #     # text_negative_ids = Variable(torch.LongTensor([torch.where(argmin_index < j, argmin_index, argmin_index + 1)argmin(cosine_distance(img_anchor,text_embedding[j for j in range(len(ids)) and j!=x ]))  for x in ids])) #neg_ids # hard negative sampling
    #     # text_negative_ids = Variable(torch.LongTensor([torch.argsort(cosine_distance(img_anchor,text_embedding[j for j in range(len(ids)) and j!=x ]))  for x in ids])) #neg_ids # hard negative sampling

    #     # image_negative_ids = Variable(torch.LongTensor([cosine_distance(for j!=x for x in ids])) #neg_ids # hard negative sampling
        
    #     # image_negative_ids = self.hard_negative_sampler(text_anchor,img_embedding,ids)
    #     # text_negative_ids = self.hard_negative_sampler(img_anchor,text_embedding,ids)
    #     negative_ids = Variable(torch.LongTensor([np.random.choice(ids[ids!=x]) for x in ids])) #neg_ids
    #     text_negative = text_embedding[negative_ids]
    #     img_negative = img_embedding[negative_ids]
    #     location_negative = locations[negative_ids]
        
    #     # else:
    #     #     text_negative = torch.zeros(batch_size).unsqueeze(0).cuda()
            
    #     #     img_negative = torch.zeros(batch_size).unsqueeze(0).cuda()
    #     #     location_negative = torch.zeros(batch_size).unsqueeze(0).cuda()
    #     # print("anchor_iddsssss",anchor_ids,negative_ids,positive_ids)
    #     # img_anchor = img_embedding[anchor_ids]
    #     text_positive = text_embedding[positive_ids]
    #     location_anchor = locations[anchor_ids]
    #     # location_positive = locations[positive_ids]
        

    #     # text_anchor = text_embedding[anchor_ids]
    #     img_positive = img_embedding[positive_ids]
    #     # print("longgggggg",negative_ids.shape)
    #     # print("location an", location_anchor,location_negative)
        
    #     # weights = [1 if location_anchor[i] == location_negative[i] else 0.5 for i in range(len(location_anchor))]
    #     weights = [1  for i in range(len(location_anchor))]
    #     margin = [0.2 if location_anchor[i] == location_negative[i] else 0.4 for i in range(len(location_anchor))]
    #     weights = torch.Tensor(margin)
    #     weights = weights.cuda()
        
    #     # i2t_triplet_loss = weights*triplet_loss_with_cosine_distance(img_anchor, text_positive, text_negative,
    #     #                                                     margin=self.margin)
    #     # t2i_triplet_loss = weights*triplet_loss_with_cosine_distance(text_anchor, img_positive, img_negative,
    #     #                                                     margin=self.margin)

    #     ######$$$$$$$
    #     i2t_triplet_loss = triplet_loss_with_cosine_distance(img_anchor, text_positive, text_negative,weights,
    #                                                         margin=weights)#self.margin)
    #     t2i_triplet_loss = triplet_loss_with_cosine_distance(text_anchor, img_positive, img_negative,weights,
    #                                                         margin=weights)#self.margin)
        
    #     i2t_cosine , t2i_cosine = cosine_distance(img_embedding,text_embedding)[1] , cosine_distance(text_embedding,img_embedding)[1]
    #     return i2t_triplet_loss.mean()+t2i_triplet_loss.mean(),i2t_cosine,t2i_cosine


class local_entropy(nn.Module):
    def __init__(self, margin=0.25):
        super(local_entropy, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, labels):#(self,cosine_similarity,labels):#
        # cosine_similarity = F.cosine_similarity(embedding1, embedding2)
        # print("heyyy")
        
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        cosine_similarity=w12 / (w1 * w2)
        # if batch_size!=1:
        #     negative_ids = Variable(torch.LongTensor([np.random.choice(ids[ids!=x]) for x in ids])) #neg_ids
        #     text_negative = text_embedding[negative_ids]
        #     img_negative = img_embedding[negative_ids]
        # else:
        #     text_negative = torch.zeros(batch_size).unsqueeze(0).cuda()
            
        #     img_negative = torch.zeros(batch_size).unsqueeze(0).cuda()
            
        # img_anchor = img_embedding[anchor_ids]
        # text_positive = text_embedding[positive_ids]
        

        # text_anchor = text_embedding[anchor_ids]
        # img_positive = img_embedding[positive_ids]
        

        # i2t_triplet_loss = triplet_loss_with_cosine_distance(img_anchor, text_positive, text_negative,
        #                                                     margin=self.margin)
        # t2i_triplet_loss = triplet_loss_with_cosine_distance(text_anchor, img_positive, img_negative,
                                                            # margin=self.margin)
        # i2t_cosine , t2i_cosine = cosine_distance(img_embedding,text_embedding)[1] , cosine_distance(text_embedding,img_embedding)[1]
        loss0 = nn.CrossEntropyLoss()(cosine_similarity, labels)  # labels: arange(batch_size)
        loss1 = nn.CrossEntropyLoss()(cosine_similarity.transpose(0, 1) , labels)
        return loss0+loss1#,i2t_cosine,t2i_cosine
        

class ContrastiveLoss_cosine3(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss_cosine3, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, labels):#(self,cosine_similarity,labels):#
        # cosine_similarity = F.cosine_similarity(embedding1, embedding2)
        # print("heyyy")
        
        batch_size = embedding1.shape[0]
        eps=1e-8
        temp3=0.5#0.5#10.0
        # embedding1 = embedding1.unsqueeze(0)
        # embedding2 = embedding2.unsqueeze(0)
        embedding1 = embedding1.unsqueeze(0)
        embedding2 = embedding2.unsqueeze(0)

        positive_pairs = torch.eye(embedding1.size(0))  # Positive pairs have 1 on diagonal
        negative_pairs = 1 - positive_pairs
        
        embedding1_norm = torch.norm(embedding1, 2, dim=2, keepdim=True)
        embedding2_norm = torch.norm(embedding2, 2, dim=2, keepdim=True)

        cosine_similarity = torch.bmm(embedding1, embedding2.transpose(1, 2))
        norm0 = torch.bmm(embedding1_norm, embedding2_norm.transpose(1, 2))
        cosine_similarity = cosine_similarity / norm0.clamp(min=eps) * temp3
        cosine_similarity = cosine_similarity.squeeze()
        # labells=(torch.eye(cosine_similarity.size(0)) > 0.5).float()
        # # print("++++",labels)
        # labells=labells.cuda()
        # loss_contrastive_1 = torch.mean((1 - labels) * torch.pow(1 - cosine_similarity, 2) +
        #                       (labels) * torch.pow(torch.clamp(cosine_similarity - self.margin, min=0.0), 2))
        loss_contrastive_1 = 0.004*torch.mean(1*(1-labels) * torch.pow(cosine_similarity, 2) +
                              0.0625*(labels) * torch.pow(torch.clamp(-cosine_similarity +self.margin, min=0.0), 2)) #0.0625*
        cosine_similarity_t = cosine_similarity.T
        # loss_contrastive_2 = torch.mean((1 - labels) * torch.pow(1 - cosine_similarity_t, 2) +
        #                       (labels) * torch.pow(torch.clamp(cosine_similarity_t - self.margin, min=0.0), 2))
        # loss_contrastive_2 = torch.mean((1 - labels) * torch.pow(1 - cosine_similarity_t, 2) +
        #                       (labels) * torch.pow(torch.clamp(cosine_similarity_t - self.margin, min=0.0), 2))
        # loss_contrastive_2 = torch.mean((1-labels) * torch.pow(cosine_similarity_t, 2) +
        #                       (labels) * torch.pow(torch.clamp(-cosine_similarity_t +self.margin, min=0.0), 2))
        loss_contrastive=loss_contrastive_1#+loss_contrastive_2
        # labels = Variable(torch.LongTensor(range(cosine_similarity.shape[0]))).cuda()
        
        # loss_contrastive_1=nn.CrossEntropyLoss()(cosine_similarity,labels)
        # loss_contrastive_2=nn.CrossEntropyLoss()(cosine_similarity.transpose(0,1),labels)
        # loss_contrastive=loss_contrastive_1+loss_contrastive_2
        loss=SelfSupervisedLoss(losses.TripletMarginLoss(distance = CosineSimilarity()))#losses.ContrastiveLoss(pos_margin=1, neg_margin=-1,distance=distances.CosineSimilarity(**kwargs), **kwargs)
        loss_contrastive=loss(embedding1,embedding2)

        
        labels = Variable(torch.LongTensor(range(batch_size))).to(embedding1.device)
        if batch_size==1:
            cosine_similarity=cosine_similarity.unsqueeze(0)
            loss_contrastive=loss(embedding1,embedding2)
            print("H")
        else:
            loss0 = nn.CrossEntropyLoss()(cosine_similarity, labels)
            loss1 = nn.CrossEntropyLoss()(cosine_similarity_t,labels) 
            loss_contrastive=loss0+loss1
        # loss_con=NTXentLoss(temperature=0.07, **kwargs)
        # loss_contrastive = loss_con()
        return loss_contrastive,cosine_similarity,cosine_similarity_t




class ContrastiveLoss_cosine(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss_cosine, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, labels):#(self,cosine_similarity,labels):#
        # cosine_similarity = F.cosine_similarity(embedding1, embedding2)
        eps=1e-8
        temp3=10#10.0
        # embedding1 = embedding1.unsqueeze(0)
        # embedding2 = embedding2.unsqueeze(0)
        embedding1 = embedding1.unsqueeze(0)
        embedding2 = embedding2.unsqueeze(0)

        positive_pairs = torch.eye(embedding1.size(0))  # Positive pairs have 1 on diagonal
        negative_pairs = 1 - positive_pairs
        
        embedding1_norm = torch.norm(embedding1, 2, dim=2, keepdim=True)
        embedding2_norm = torch.norm(embedding2, 2, dim=2, keepdim=True)

        cosine_similarity = torch.bmm(embedding1, embedding2.transpose(1, 2))
        norm0 = torch.bmm(embedding1_norm, embedding2_norm.transpose(1, 2))
        cosine_similarity = cosine_similarity / norm0.clamp(min=eps) * temp3
        cosine_similarity = cosine_similarity.squeeze()
        # labels=(torch.eye(cosine_similarity.size(0)) > 0.5).float()
        # # print("++++",labels)
        # labels=labels.cuda()
        loss_contrastive_1 = 0.004*torch.mean((1 - labels) * torch.pow(1 - torch.sqrt(torch.pow(cosine_similarity,2)), 2) +
                              0.0625*(labels) * torch.pow(torch.clamp(cosine_similarity, min=0.0), 2))
        # loss_contrastive_1 = torch.mean(1*(1-labels) * torch.pow(cosine_similarity, 2) +
        #                       (labels) * torch.pow(torch.clamp(-cosine_similarity +self.margin, min=0.0), 2)) #0.0625*
        cosine_similarity_t = cosine_similarity.T
        # loss_contrastive_2 = torch.mean((1 - labels) * torch.pow(1 - cosine_similarity_t, 2) +
        #                       (labels) * torch.pow(torch.clamp(cosine_similarity_t - self.margin, min=0.0), 2))
        # loss_contrastive_2 = torch.mean((1 - labels) * torch.pow(1 - cosine_similarity_t, 2) +
        #                       (labels) * torch.pow(torch.clamp(cosine_similarity_t - self.margin, min=0.0), 2))
        # loss_contrastive_2 = torch.mean((1-labels) * torch.pow(cosine_similarity_t, 2) +
        #                       (labels) * torch.pow(torch.clamp(-cosine_similarity_t +self.margin, min=0.0), 2))
        loss_contrastive_2 = 0.004*torch.mean((1 - labels) * torch.pow(1 - torch.sqrt(torch.pow(cosine_similarity_t,2)), 2) +
                              0.0625*(labels) * torch.pow(torch.clamp(cosine_similarity_t, min=0.0), 2))
        loss_contrastive=loss_contrastive_1+loss_contrastive_2
        # labels = Variable(torch.LongTensor(range(cosine_similarity.shape[0]))).cuda()
        
        # loss_contrastive_1=nn.CrossEntropyLoss()(cosine_similarity,labels)
        # loss_contrastive_2=nn.CrossEntropyLoss()(cosine_similarity.transpose(0,1),labels)
        # loss_contrastive=loss_contrastive_1+loss_contrastive_2
        return loss_contrastive,cosine_similarity,cosine_similarity_t

# class ContrastiveLoss_cosine(nn.Module):
#     def __init__(self, margin):
#         super(ContrastiveLoss_cosine, self).__init__()
#         self.margin = margin
    
#     def forward(self, embedding1, embedding2, labels):#(self,cosine_similarity,labels):#
#         # cosine_similarity = F.cosine_similarity(embedding1, embedding2)
#         eps=1e-8
#         temp3=10#10.0
#         # embedding1 = embedding1.unsqueeze(0)
#         # embedding2 = embedding2.unsqueeze(0)
#         embedding1 = embedding1.unsqueeze(1)
#         embedding2 = embedding2.unsqueeze(0)
#         # embedding1_norm = torch.norm(embedding1, 2, dim=2, keepdim=True)
#         # embedding2_norm = torch.norm(embedding2, 2, dim=2, keepdim=True)

#         # cosine_similarity = torch.bmm(embedding1, embedding2.transpose(1, 2))
#         # norm0 = torch.bmm(embedding1_norm, embedding2_norm.transpose(1, 2))
#         # cosine_similarity = cosine_similarity / norm0.clamp(min=eps) * temp3
#         cosine_similarity = F.cosine_similarity(embedding1, embedding2,dim=2)/temp3
#         cosine_similarity = cosine_similarity.squeeze()
#         # print("today",cosine_similarity.shape)
#         loss_contrastive_1 = torch.mean((1 - labels) * torch.pow(1 - cosine_similarity, 2) +
#                               (labels) * torch.pow(torch.clamp(cosine_similarity - self.margin, min=0.0), 2))
#         cosine_similarity_t = cosine_similarity.T
#         loss_contrastive_2 = torch.mean((1 - labels) * torch.pow(1 - cosine_similarity_t, 2) +
#                               (labels) * torch.pow(torch.clamp(cosine_similarity_t - self.margin, min=0.0), 2))
#         loss_contrastive=loss_contrastive_1+loss_contrastive_2
#         return loss_contrastive,cosine_similarity,cosine_similarity_t
# class ContrastiveLoss_cosine(nn.Module):
#     def __init__(self, margin):
#         super(ContrastiveLoss_cosine, self).__init__()
#         self.margin = margin
    
#     def forward(self, embedding1, embedding2, labels):#(self,cosine_similarity,labels):#
#         eps=1e-8
#         temp3=0.07#10.0
#         embedding1 = embedding1.unsqueeze(0)
#         embedding2 = embedding2.unsqueeze(0)
#         embedding1_norm = torch.norm(embedding1, 2, dim=2, keepdim=True)
#         embedding2_norm = torch.norm(embedding2, 2, dim=2, keepdim=True)

#         cosine_similarity = torch.bmm(embedding1, embedding2.transpose(1, 2))
#         norm0 = torch.bmm(embedding1_norm, embedding2_norm.transpose(1, 2))
#         cosine_similarity = cosine_similarity / norm0.clamp(min=eps) * temp3
#         cosine_similarity = cosine_similarity.squeeze()
        
#         cosine_similarity_t = cosine_similarity.T
#         loss0 = F.cross_entropy(cosine_similarity, labels)
#         loss1 = F.cross_entropy(cosine_similarity_t, labels)
#         loss_contrastive = loss0 + loss1

#         # scores = embedding1.mm(embedding2.t())
#         # scores /= self.hparams.softmax_temperature
#         # scores1 = scores.transpose(0, 1)


#         # cosine_similarity = torch.bmm(embedding1, embedding2.transpose(1, 2))
#         # norm0 = torch.bmm(embedding1_norm, embedding2_norm.transpose(1, 2))
#         # cosine_similarity = cosine_similarity / norm0.clamp(min=eps) * temp3
#         # cosine_similarity = cosine_similarity.squeeze()
        
#         # loss_contrastive_1 = torch.mean((1 - labels) * torch.pow(1 - cosine_similarity, 2) +
#         #                       (labels) * torch.pow(torch.clamp(cosine_similarity - self.margin, min=0.0), 2))
#         # cosine_similarity_t = cosine_similarity.T
#         # loss_contrastive_2 = torch.mean((1 - labels) * torch.pow(1 - cosine_similarity_t, 2) +
#         #                       (labels) * torch.pow(torch.clamp(cosine_similarity_t - self.margin, min=0.0), 2))
#         # loss_contrastive=loss_contrastive_1+loss_contrastive_2
#         return loss_contrastive,cosine_similarity,cosine_similarity_t



# class ContrastiveLoss_euclidean(nn.Module):
#     def __init__(self, margin):
#         super(ContrastiveLoss_euclidean, self).__init__()
#         self.margin = margin
    
#     def forward(self,embedding1, embedding2, labels):#(self, embedding1, embedding2, labels):
#         # cosine_similarity = F.cosine_similarity(embedding1, embedding2)
#         embedding1 = embedding1.unsqueeze(0)
#         embedding2 = embedding2.unsqueeze(0)
#         euclidean_distance = torch.nn.functional.pairwise_distance(embedding1, embedding2, keepdim=True)
#         print("eucl",euclidean_distance.shape)
#         euclidean_distance_t = euclidean_distance.T
#         loss_contrastive = torch.mean((1 - labels) * torch.pow(euclidean_distance, 2) +
#                                       (labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#         return loss_contrastive,euclidean_distance,euclidean_distance_t
class ContrastiveLoss_euclidean(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss_euclidean, self).__init__()
        self.margin = margin
    
    def forward(self,embedding1, embedding2, labels):#(self, embedding1, embedding2, labels):
        # cosine_similarity = F.cosine_similarity(embedding1, embedding2)
        embedding1 = embedding1.unsqueeze(1)
        embedding2 = embedding2.unsqueeze(0)
        euclidean_distance = torch.sqrt(torch.sum((embedding1 - embedding2)**2, dim=2))
        # print("eucl",euclidean_distance.shape)
        # euclidean_distance=euclidean_distance.squeeze()
        euclidean_distance_t = euclidean_distance.T
        loss_contrastive = torch.mean((1 - labels) * torch.pow(euclidean_distance, 2) +
                                      (labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive,euclidean_distance,euclidean_distance_t

class ContrastiveLoss_euclidean2(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss_euclidean, self).__init__()
        self.margin = margin
    
    def forward(self,distance, labels):#(self, embedding1, embedding2, labels):
        # cosine_similarity = F.cosine_similarity(embedding1, embedding2)
        # euclidean_distance = torch.nn.functional.pairwise_distance(embedding1, embedding2, keepdim=True)
        loss_contrastive = torch.mean((1 - labels) * torch.pow(distance, 2) +
                                      (labels) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive


def local_loss(
    img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):

    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = cap_lens[i]  
    
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 949]
        word = word.repeat(batch_size, 1, 1)  # [4, 768, 949]
        context = img_features  # [48, 768, 19, 19]

        weiContext, attn = attention_fn(
            word, context, temp1
        )  # [48, 768, 25], [48, 25, 19, 19]

        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [25, 19, 19]
        word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        word = word.view(batch_size * words_num, -1)  # [1200, 768]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)  # [48, 48]

    labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

    loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return loss0, loss1, att_maps 

# def cosine_similarity(x1, x2, dim=1, eps=1e-8):
#     """Returns cosine similarity between x1 and x2, computed along dim."""
#     w12 = torch.sum(x1 * x2, dim)
#     w1 = torch.norm(x1, 2, dim)
#     w2 = torch.norm(x2, 2, dim)
#     return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def cosine_similarity(embedding1, embedding2, dim=1, eps=1e-8,temp3=0.5):
    #forglobal_local
    """Returns cosine similarity between x1 and x2, computed along dim."""
    # print("embedding 1 and 2",embedding1.shape,embedding2.shape)
    embedding1=embedding1.unsqueeze(0)
    embedding2=embedding2.unsqueeze(0)
    embedding1_norm = torch.norm(embedding1, 2, dim=2, keepdim=True)
    embedding2_norm = torch.norm(embedding2, 2, dim=2, keepdim=True)

    cosine_similarity = torch.bmm(embedding1, embedding2.transpose(1, 2))
    norm0 = torch.bmm(embedding1_norm, embedding2_norm.transpose(1, 2))
    cosine_similarity = cosine_similarity / norm0.clamp(min=eps) * temp3
    cosine_similarity = cosine_similarity.squeeze()
        
    return cosine_similarity


# def cosine_similarity(embedding1, embedding2, dim=1, eps=1e-8,temp3=0.5):
#     """Returns cosine similarity between x1 and x2, computed along dim."""
#     # print("embedding 1 and 2",embedding1.shape,embedding2.shape)
#     # embedding1=embedding1.unsqueeze(0)
#     # embedding2=embedding2.unsqueeze(0)
    
#     print("embeddingsss",embedding1.shape,embedding2.shape)
#     embedding1_norm = torch.norm(embedding1, 2, dim=2, keepdim=True)
#     embedding2_norm = torch.norm(embedding2, 2, dim=2, keepdim=True)

#     cosine_similarity = torch.bmm(embedding1, embedding2.transpose(2,3))
#     norm0 = torch.bmm(embedding1_norm, embedding2_norm.transpose(2,3))
#     cosine_similarity = cosine_similarity / norm0.clamp(min=eps) * temp3
#     # cosine_similarity = cosine_similarity.squeeze()
        
#     # cosine_similarity=F.cosine_similarity(embedding2.unsqueeze(2), embedding1.unsqueeze(1), dim=3)
#     print("cosine",cosine_similarity.shape)
#     return cosine_similarity
# def cosine_similarity(x1, x2, dim=1, eps=1e-8):
#     """Returns cosine similarity between x1 and x2, computed along dim."""
#     w12 = torch.bmm(x1,x2)
#     w1 = torch.norm(x1, 2, dim)
#     w2 = torch.norm(x2, 2, dim)
#     return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

# def cosine_similarity(x1, x2, dim=1, eps=1e-8):
#     """Returns cosine similarity between x1 and x2, computed along dim."""
#     w12 = torch.bmm(
#             x1, x2) #/ self.hparams.local_temperature.transpose(1, 2)
#     w1 = torch.norm(x1, 2, dim=2)
#     w2 = torch.norm(x2, 2, dim=2)
#     return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
def cosine_similarity2(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def local_contrastive_loss3(
    img_features, word_embedds,weiContext, cap_lens, w_at,margin,temp1=4.0, temp2=5.0, temp3=0.5, agg="sum"
):
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
    c_loss = ContrastiveLoss_cosine2(margin=margin)
    loss0,_,_=c_loss(word_embedds, weiContext,labels)
    # print("wwwwwwwww",w_at.view(-1).shape,c_loss(word_embedds, weiContext,labels).shape)
    # print(w_at)
    # loss0=torch.sum(c_loss(word_embedds, weiContext,labels)[0]* w_at.view(-1))/ batch_size
    # loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return loss0#+loss1#, loss1, att_maps 

def local_contrastive_loss2(
    img_features, word_embedds,weiContext, cap_lens,locations,margin,temp1=4.0, temp2=5.0, temp3=0.5, agg="sum"
):
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



# @staticmethod
def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
    ''' Compute the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)
        #######
        if batch_size==1:
            output=output.unsqueeze(0)
        ########
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print("heyyy",output.shape,target.shape,target.view(1, -1).shape)
        correct = pred.eq(target)#.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res[0],correct_k,batch_size







def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
    ''' Compute the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print("heyyy",output.shape,target.shape,target.view(1, -1).shape)
        correct = pred.eq(target)#.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res[0],correct_k,batch_size



def precision_at_k(cosine_similarity, k=1):
    with torch.no_grad():
        _, top_indices = cosine_similarity.topk(k, largest=True)
        # print("cosine",cosine_similarity.shape)
        # print("cosine",top_indices)
        # print("cosine2",cosine_similarity.shape)
        batch_size=cosine_similarity.shape[0]
        # print("labels",torch.arange(batch_size).view(-1, 1))
        # print("top",top_indices)
        true_matches = top_indices == torch.arange(batch_size).view(-1, 1).cuda()
        precision = true_matches.float().mean().item()
        # print("first precision",precision)
        # print("true_matches",true_matches.float().sum())
        true=true_matches.float().sum()
        
        return precision,true,batch_size



# def precision_at_k(cosine_similarity, k=(1,)):
#     # print("sss",cosine_similarity)
#     # print("cosine",cosine_similarity)
#     # print("cosine",cosine_similarity)
#     # print(cosine_similarity.dim())
    
#     # print("here cosine",cosine_similarity.shape)
#     with torch.no_grad():
#         # if cosine_similarity.dim()>1:
#         #     # print("hey",cosine_similarity.shape)
#         #     # print("coooosine",cosine_similarity.dim(),cosine_similarity.shape)
#         #     _, top_indices = cosine_similarity.topk(k, largest=True,dim=1)
#         #     batch_size=cosine_similarity.shape[0]
#         # else:
#         #     cosine_similarity=cosine_similarity.unsqueeze(0)
#         #     _, top_indices = cosine_similarity,0
#         #     batch_size=1
#         maxk = max(k)
#         _, top_indices = cosine_similarity.topk(maxk, largest=True)#,dim=1)
#         # print("cosine",top_indices)
#         # print("cosine2",cosine_similarity.shape)
#         # print("salam")
#         # print(_)
        
#         batch_size=cosine_similarity.shape[0]
#         # print("labels",torch.arange(batch_size).view(-1, 1))
#         # print(torch.arange(batch_size).view(-1, 1))
#         print("batch_size",top_indices,torch.arange(batch_size).view(-1, 1))
#         true_matches = top_indices == torch.arange(batch_size).view(-1, 1).cuda()
#         precision = true_matches.float()
#         # print("first precision",precision)
#         # print("true_matches",true_matches.float().sum())
#         true=true_matches.float().sum()
#         # print(true_matches)
#         return precision.mean(),true,batch_size




# def cosine_distance(x1, x2, dim=1, eps=1e-8):
#     """Returns (1 - cosine similarity) between x1 and x2, computed along dim.
#     """
#     w12 = torch.sum(x1 * x2, dim)
#     w1 = torch.norm(x1, 2, dim)
#     w2 = torch.norm(x2, 2, dim)
    
#     return 1 - (w12 / (w1 * w2).clamp(min=eps)).squeeze()



# def triplet_loss_with_cosine_distance(anc, pos, neg, margin=0.5):
#     score = cosine_similarity(anc, pos) - cosine_similarity(anc, neg) + margin
#     #     z = torch.zeros_like(score)
#     relu=nn.ReLU()
#     return relu(score)


# def sent_triplet_loss(cnn_code, rnn_code, labells, batch_size=16):
#     b_size=cnn_code.shape[0]
#     ids = np.array(list(range(b_size)))
#     neg_ids = Variable(torch.LongTensor([np.random.choice(ids[ids!=x]) for x in ids])) 
    
#     co_score=cosine_similarity(cnn_code,rnn_code)
#     # print("cosine_",co_score.shape)
    
#     labels=Variable(torch.LongTensor(range(b_size))) 
#     anchor_ids = labels
#     positive_ids = labels
#     negative_ids = neg_ids
#     # print("anchor",anchor_ids.shape)
#     # print("code",cnn_code.shape)
#     img_anchor = cnn_code[anchor_ids]
#     text_positive = rnn_code[positive_ids]
#     text_negative = rnn_code[negative_ids]

#     text_anchor = rnn_code[anchor_ids]
#     img_positive = cnn_code[positive_ids]
#     img_negative = cnn_code[negative_ids]

#     i2t_triplet_loss = triplet_loss_with_cosine_distance(img_anchor, text_positive, text_negative,
#                                                          margin=0.5).mean()
#     t2i_triplet_loss = triplet_loss_with_cosine_distance(text_anchor, img_positive, img_negative,
#                                                          margin=0.5).mean()

#     return i2t_triplet_loss+ t2i_triplet_loss,co_score,co_score.T





# class sent_triplet_loss(nn.Module):
#     """Compute contrastive loss"""

#     def __init__(self, margin=0, measure=False, max_violation=False):
#         super(sent_triplet_loss, self).__init__()
#         self.margin = margin
#         self.max_violation = max_violation

#     def sim(self, im, s):
#         """Cosine similarity between all the image and sentence pairs"""
#         return im.mm(s.t())

#     def forward(self, im, s):
#         # compute image-sentence score matrix
#         scores = self.sim(im, s)
#         diagonal = scores.diag().view(im.size(0), 1)
#         d1 = diagonal.expand_as(scores)
#         d2 = diagonal.t().expand_as(scores)

#         # compare every diagonal score to scores in its column
#         # caption retrieval
#         cost_s = (self.margin + scores - d1).clamp(min=0)
#         # compare every diagonal score to scores in its row
#         # image retrieval
#         cost_im = (self.margin + scores - d2).clamp(min=0)

#         # clear diagonals
#         mask = torch.eye(scores.size(0)) > 0.5
#         I = Variable(mask)
#         if torch.cuda.is_available():
#             I = I.cuda()
#         cost_s = cost_s.masked_fill_(I, 0)
#         cost_im = cost_im.masked_fill_(I, 0)

#         # keep the maximum violating negative for each query
#         if self.max_violation:
#             cost_s = cost_s.max(1)[0]
#             cost_im = cost_im.max(0)[0]

#         return cost_s.sum() + cost_im.sum(),scores,scores.T


# class sent_triplet_loss(nn.Module):
#     def __init__(self, nmax=1, margin=0.2):
#         super(sent_triplet_loss, self).__init__()
#         self.margin = margin
#         self.nmax = nmax

#     def forward(self, imgs, caps):
#         scores = torch.mm(imgs, caps.t())
#         sim=scores
#         diag = scores.diag()

#         # Reducing the score on diagonal so there are not selected as hard negative
#         scores = scores - 2 * torch.diag(scores.diag())

#         sorted_cap, _ = torch.sort(scores, 0, descending=True)
#         sorted_img, _ = torch.sort(scores, 1, descending=True)

#         # Selecting the nmax hardest negative examples
#         max_c = sorted_cap[: self.nmax, :]
#         max_i = sorted_img[:, : self.nmax]

#         # Margin based loss with hard negative instead of random negative
#         neg_cap = torch.sum(
#             torch.clamp(
#                 max_c + (self.margin - diag).view(1, -1).expand_as(max_c), min=0
#             )
#         )
#         neg_img = torch.sum(
#             torch.clamp(
#                 max_i + (self.margin - diag).view(-1, 1).expand_as(max_i), min=0
#             )
#         )

#         loss = neg_cap + neg_img

#         return loss,sim,sim.T

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

def local_loss(
    img_features, words_emb, cap_lens,locations, margin=0.25,temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):

    # patch_local_atten_layer = nn.MultiheadAttention(
    #         256, 1, batch_first=True).cuda()
    batch_size = img_features.shape[0]

    att_maps = []
    wei_list = []
    words = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = cap_lens[i]  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
        # print("words_num",words_num)
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
        word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
        context = img_features  # [48, 768, 19, 19]

        # weiContext, attn = attention_fn(
        #     word, context, temp1
        # )  # [48, 768, 25], [48, 25, 19, 19]
        
        # mask_pad = torch.from_numpy((np.array(sents)[:, 1:] == "<pad>")|(np.array(sents)[:, 1:] =='<s>')|(np.array(sents)[:, 1:] =='</s>')|(np.array(sents)[:, 1:] =='<mask>')|(np.array(sents)[:, 1:] =='unk')|(np.array(sents)[:, 1:] =='')|(np.array(sents)[:, 1:] =='Ċ')).type_as(image).bool()
        # print("111",context.shape,word.shape)
        crossAttention = CrossAttention(512,"txt_img")#,sents)
        weiContext, attn,_ = crossAttention(
            context,word)
          # [48, 768, 25], [48, 25, 19, 19]
        # att_maps.append(
        #     attn[i].unsqueeze(0).contiguous()
        # )  # add attention for curr index  [25, 19, 19]
        # word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
        # weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        # word = word.view(batch_size * words_num, -1)  # [1200, 768]
        # # print("wei",weiContext.shape)
        # weiContext = weiContext.view(batch_size * 2250, -1)  # [1200, 768]
        words.append(word)
        wei_list.append(weiContext)
        # row_sim = cosine_similarity(word, weiContext)
        # row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

        # row_sim.mul_(temp2).exp_()
        # if agg == "sum":
        #     row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        # else:
        #     row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
        # row_sim = torch.log(row_sim)

        # similarities.append(row_sim)

    # similarities = torch.cat(similarities, 1)  #
    # similarities = similarities * temp3
    # similarities1 = similarities.transpose(0, 1)  # [48, 48]
    wei_list = torch.cat(wei_list, dim=0) 
    words = torch.cat(words, dim=0) 
    wei_list=wei_list.reshape(batch_size,-1)
    words=words.reshape(batch_size,-1)
    
    # wei_list=[]
    # words=[]
    # labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

    # loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    # loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    labs = torch.arange(batch_size).long() #global
    labels = torch.eye(batch_size)[labs].cuda() #global
    c_loss = ContrastiveLoss_cosine2(margin=margin,mode="local")
    loss0,_,_=c_loss(words, wei_list,labels,locations)
    return loss0#, loss1, att_maps
