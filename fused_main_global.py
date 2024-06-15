import os
import sys
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader,SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# from imgaug import augmenters as iaa
from datetime import datetime
from captum.attr import LayerGradCam,GuidedGradCam
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from fused_model import image_text_attention_prototype,image_text_attention_test,image_text_attention,image_text_attention_global,SelfAttention, DepthAttention, LocalEmbedding_3d
from fused_dataset import split_dataset_cv,process_excel,BertDataset#,load_data_for_patient,process_excel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.autograd import Variable
from losses import ContrastiveLoss_cosine,ContrastiveLoss_euclidean,ContrastiveLoss_cosine2
from losses import local_contrastive_loss2,cosine_similarity,cosine_distance,cosine_similarity2,local_contrastive_loss3,local_loss
#attention maps
# from medcam import medcam
#from monai.visualize import GradCAM
#import pytorch_lightning
#from pytorch_lightning import GuidedGradCam
#############from models.classifier import UnetClassifier, EditableUnet
#from utils.gradcam_utils import GradCam, visualize_gcam
#########from utils.dataset import split_dataset, EyegazeDataset, collate_fn
#from utils.utils import cyclical_lr
#from utils.dice_loss import GeneralizedDiceLoss
#from utils.visualization import plot_roc_curve
from transformers import AdamW
from sklearn.metrics import roc_auc_score, roc_curve
# from lgg_utils import get_model_name,cyclical_lr
from fused_dataset import BertDataset,split_dataset_cv,process_excel#,load_data_for_patient,process_excel
# from fused_model import generate_model
import pickle
from sklearn.manifold import TSNE
plt.rcParams['figure.figsize'] = [25, 10]
from losses import precision_at_k

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('lgg_unet')
logging.getLogger('matplotlib.font_manager').disabled = True
pil_logger = logging.getLogger('PIL').setLevel(logging.INFO)

# import torch.distributed as dist
# dist.init_process_group(backend='nccl')
def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch SickKids Brain MRI')

    # Data
    parser.add_argument('--data_path', type=str, default="/hpf/largeprojects/fkhalvati/Sara/lgg/Nomogram_study_LGG_data_Nov.27.xlsx", help='Data path')
    parser.add_argument('--image_path', type=str, default="/hpf/largeprojects/fkhalvati/Projects/SickKids_Brain_Preprocessing/preprocessed_all_seq_kk_july_2022", help='image_path')
    #parser.add_argument('--heatmaps_path', type=str, help='Heatmaps directory',
                       # default='/home/k/khalvati/agniho24/Dataset/fixation_heatmaps')
    parser.add_argument('--output_dir', type=str, default='/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text', help='Output directory')
    parser.add_argument('--class_names', type=list, default=['other', 'Pilocytic Astrocytoma', 'Ganglioglioma'], help='Label names for classification')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('--resize', type=int, default=224, help='Resizing images')

    # Training
    parser.add_argument('--batch_size', type=int, default=4, help='batch size') #4
    parser.add_argument('--num_epochs', type=int, default=800, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate') #5e-2  #1e-3 #compare:5e-6 #6e-6 (best till now) 0.0002
    parser.add_argument('--scheduler', default=False, action='store_true', help='[USE] scheduler') ##true?
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')

    ## UNET Specific arguments.
    parser.add_argument('--model_type', default='unet', choices=['baseline', 'unet'], help='baseline, unet')
    parser.add_argument('--model_teacher', type=str, default='timm-efficientnet-b0', help='model_teacher')
    parser.add_argument('--pretrained_name', type=str, default='noisy-student', help='model pretrained value')
    parser.add_argument('--dropout', type=float, default=0, help='dropout') #0.1 #0.15
    parser.add_argument('--second_loss', type=str, default='ce', choices=['dice', 'ce'], help='Segmentation loss')

    # Misc
    parser.add_argument('--gpus', type=str, default='0', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--viz', default=False, action='store_true', help='[USE] Vizdom')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test [same as train if not set]')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--margin', type=float, default=0.25)
    parser.add_argument('--similarity_measure', type=str, default='euclidian')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--global_local', type=str, default='global')
    parser.add_argument('--accumulate', type=float , default=32)

    return parser

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

random_seed(14, True)

splits=KFold(n_splits=5,shuffle=True,random_state=42)



def count_trainable_parameters(model):
    component_params = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            component_name = name.split('.')[0]
            num_params = param.numel()
            
            if component_name not in component_params:
                component_params[component_name] = 0
            component_params[component_name] += num_params
            total_params += num_params
    
    return component_params, total_params



def load_data(train_dataset,valid_dataset,test_dataset, batch_size):
      
    train_dl= DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return train_dl, valid_dl, test_dl
   
class DynamicWeightedLoss(nn.Module):
    def __init__(self, num_losses):
        super(DynamicWeightedLoss, self).__init__()
        self.num_losses = num_losses
        self.weights = nn.Parameter(torch.ones(num_losses))

    def forward(self, losses):
        weighted_losses = losses * self.weights.unsqueeze(0).cuda()
        total_loss = torch.sum(weighted_losses)
        return total_loss , self.weights



def train_global_model(args, data_ds,test_dl, output_model_path, tuning=False):
    
    global_step = 0
    temperature=0.1
    softmax_temperature= 0.07
    local_temperature=0.1
   
    lambda_1: float =1
    lambda_2: float =1
    
    lambda_3: float = 0.125
    num_heads = 1
    emb_dim = 128
    
 #   optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
    batch_size=args.batch_size
    num_batch_accumulate = args.accumulate/batch_size #128 / batch_size
 
    classifier_criterion = nn.BCEWithLogitsLoss()#BCELoss()#.
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
#    optimizer.zero_grad()
#    for t in range(n_trial):
    weight_path="/hpf/largeprojects/fkhalvati/Sara/pretrain/resnet_18_23dataset.pth"
    best_auc=0
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_ds)))):
    
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_dl = DataLoader(data_ds, batch_size=batch_size, sampler=train_sampler)
        valid_dl = DataLoader(data_ds, batch_size=batch_size, sampler=test_sampler)
        train_dl = DataLoader(whole_data, batch_size=batch_size,shuffle=True)
        print("lenn",len(train_dl))
        model=image_text_attention_global(emb_dim=128,num_heads=num_heads,mode="global")
        # opt={}
        ##################3pretrain
        # model = generate_model(model_depth=18, inplanes=inplanes, n_classes=1039)
        model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        model.cnn.fc  = nn.Linear(512, 1)
        net_dict = model.cnn.state_dict()
        
        pretrain = torch.load(weight_path)
        pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.cnn.load_state_dict(net_dict)
        ##############3pretrain
        # model.load_state_dict(weight_path['state_dict'])  ####mednet
        # optimizer.load_state_dict(weight_path['optimizer']) ### mednet
        # for p in parameters:
        for pname, p in model.cnn.named_parameters():
            p.requires_grad = False
        # for pname, p in model.cnn.conv1.named_parameters():
        #     p.requires_grad = True
        # for pname, p in model.cnn.layer2.named_parameters():
        #     p.requires_grad = True
        for p in model.cnn.layer4.parameters():
            p.requires_grad = True
        for p in model.cnn.layer3.parameters():
            p.requires_grad = True
        for p in model.cnn.fc.parameters():
            p.requires_grad = False
        # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # patch_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # word_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # model.cnn.fc  = nn.Linear(512, 1)#nn.Sequential(nn.Linear(512, 256),nn.Linear(256, 1))#3) #512
        device=torch.device("cuda:0")
        # model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)#,device_ids = [int(i) for i in "0,1".split(',')])
        model.to(device)
        # print("gpu_id",torch.cuda.get_device_name(1),torch.cuda.get_device_name(0))
        # model.module.to(torch.device('cuda:1'))
        # patch_local_atten_layer.to(device)
        # word_local_atten_layer.to(device)
        
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=400,T_mult=1, eta_min=1e-8, last_epoch=-1)
            
        #(optimizer, step_size=10, gamma=0.1)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        optimizer.zero_grad()
        ita_list=[]
        ita_list_val=[]
        local_list=[]
        cl_list=[]
        for epoch in range(args.num_epochs):
            model.train()
            # model.cnn.eval()
            # # model.cnn.conv1.trian()
            # model.cnn.layer4.train()
            # model.cnn.layer3.train()
            train_loss = 0
            counter = 0
            num_batches=0
            num_batches_valid=0
            val_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]
     #       print(len(train_dl))
            i2t_corr_tr=0
            batch_epoch_tr=0
            i2t_corr_val=0
            t2i_corr_tr=0
            t2i_corr_val=0
            batch_epoch_val=0
            #******
            # image_embeddings = []
            
            report_embeddings = []
            image_embeddings=[]
            label_list=[]
            pred_list=[]
            # image_embeddings_1 = torch.Tensor().cuda()
            # image_embeddings_2= torch.Tensor().cuda()
            # image_embeddings = torch.Tensor().cuda()
            # report_embeddings = torch.Tensor().cuda()
            
            for num_batches,(images,text, labells,masks) in enumerate(train_dl): 
                # print("HI")
                # print("report",len(report_embeddings))
                device="cuda:0"#torch.device("cuda:0")
                images= images.to(device)#,labells.to(device)
                labells= labells.to(device)
                mask = text['attention_mask'].to(device)
                input_id = text['input_ids'].squeeze(1).to(device)

 
    #            loss_classifier = classifier_criterion(y_pred, labels)

    #            epoch_loss += total_loss.item()

                
                # model.to(device)
                # optimizer.zero_grad()
                # pred=model(images)
                images=images.float()
                # pred,img_emb_q,report_emb_q,patch_emb_q,word_emb_q,word_attn_q,sents,patch_atten_output,word_atten_output= model(images,input_id, mask)
                pred,img_emb_q,report_emb_q,patch_emb_q= model(images,input_id, mask)
                # print("patch_emb_q",patch_emb_q.shape)
                pred_list.append(pred)
                label_list.append(labells)
                # print("params",model.parameters)
                # del patch_emb_q,word_emb_q,word_attn_q,sents,patch_atten_output,word_atten_output
                # pred=pred.detach()
                prob = torch.sigmoid(pred)
                image_embeddings.append(img_emb_q)
                # image_embeddings = np.append(image_embeddings, img_emb_q)
                report_embeddings.append(report_emb_q)
                
                # euclidean_distance = torch.nn.functional.pairwise_distance(img_emb_q, report_emb_q, keepdim=True)
                # # torch.cuda.empty_cache()
                # # distance_list.append(euclidean_distance)
                # img_emb_q = img_emb_q.unsqueeze(1)
                # report_emb_q = report_emb_q.unsqueeze(1)
                # del pred,prob
                # if len(report_embeddings)<=4:

                #     image_embeddings_1 = torch.cat((image_embeddings_1, img_emb_q))#, dim=1)
                # else:
                #     image_embeddings_2 = torch.cat((image_embeddings_2, img_emb_q))
                # image_embeddings = torch.cat((image_embeddings, img_emb_q),dim=1)

                # # print("image",len(image_embeddings_1),len(image_embeddings_2))
                # report_embeddings = torch.cat((report_embeddings, report_emb_q), dim=1)
                # print("lennn",len(report_embeddings),len(image_embeddings))
                # print("embb",report_emb_q.shape)
                if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                    #  optimizer.zero_grad()
                    # print("Hi")
                    # distance_list = torch.cat(distance_list, dim=0)
                    # print("num_batches",num_batches)
                    
                    report_embeddings = torch.cat(report_embeddings, dim=0)
                    # print("reporttttt",report_embeddings.shape)
                    image_embeddings=torch.cat(image_embeddings, dim=0)   
                    pred_batch=torch.cat(pred_list, dim=0) 
                    label_batch=torch.cat(label_list, dim=0) 
                    # loss_classification=classifier_criterion(pred_batch,label_batch.unsqueeze(1).float())   
                    # print("imageee",image_embeddings.shape)                  
                    bz = len(report_embeddings)
                    
                    # distance_list=torch.tensor(distance_list, requires_grad=True)
                    # distance_list=distance_list.cuda()
                    labs = torch.arange(bz).type_as(report_emb_q).long() #global
                    labels = torch.eye(bz).type_as(report_emb_q)[labs] #global
                    # labels = torch.arange(bz).type_as(report_emb_q).long()
                    # labels = Variable(torch.LongTensor(range(bz))).cuda()
                    # print("labellllls",labels.shape)
                    if args.similarity_measure=="euclidian":
                        # print("ok")
                        loss_fn = ContrastiveLoss_euclidean(margin=args.margin)
                    else:
                        loss_fn =ContrastiveLoss_cosine2(margin=args.margin)
                    loss0,i_t_scores,t_i_scores = loss_fn(image_embeddings,report_embeddings,labels)
                    i_t_scores=cosine_similarity(image_embeddings,report_embeddings)
                    i2t_acc1_tr,i2t_corr_tr_batch,i2t_batch_tr = precision_at_k(i_t_scores)#, labels, top_k=(1,))
                    print("pr",i2t_acc1_tr)
                    # t2i_acc1_tr,t2i_corr_tr_batch,t2i_batch_tr  = precision_at_k(t_i_scores, labels, top_k=(1,))
                    # print("test prec",i2t_acc1_tr,t2i_acc1_tr)
                    i2t_corr_tr+=i2t_corr_tr_batch
                    # t2i_corr_tr+=t2i_corr_tr_batch
                    
                    batch_epoch_tr+=i2t_batch_tr
                    # print("training precision",(i2t_acc1_tr[0] + t2i_acc1_tr[0]) / 2.)
                    loss0.backward()


                    optimizer.step() 
                    # for element in distance_list:
                    #     # print(element.requires_grad)
                    #     print("element",element.grad)
                    optimizer.zero_grad() 
                    train_loss+=loss0.detach().item()#*batch_size ### Sajith: added detach 
                    train_batches+=1

                    # distance_list = []
                    image_embeddings = []
                    report_embeddings = []
                    pred_list=[]
                    label_list=[]
                    # image_embeddings_1 = torch.Tensor().cuda()
                    # image_embeddings = torch.Tensor().cuda()

                    # report_embeddings = torch.Tensor().cuda()
                for i in range(len(labells.tolist())):
                    # print("ll",labells)
                    
                    training_ture.append(labells.tolist()[i])#[0])
                    training_estimated.append(prob.tolist()[i])#[0]) #prob
               #     model_grad = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)

    #            corr = (pred>0.0).squeeze().long() != labells
                # train_loss+=loss.detach().item()*batch_size ### Sajith: added detach 
                #?
                # train_batches+=1
                counter += 1

            # Calculate average over epoch
            # total_train_err[epoch] = float(train_err) / total_epoch
    ##        total_train_loss[epoch] = float(train_loss) / (n + 1)
            # scheduler.step()
            train_loss = train_loss/(train_batches)#*batch_size)  
            print("loss_ita",train_loss)
            ita_list.append(train_loss)
            if epoch%10==0:
                torch.save(model.state_dict(), os.path.join(args.output_dir,f"cls_global_only__fold__{fold}__epoch__{epoch}__margin{args.margin}_similarity{args.similarity_measure}"))
            del loss0 ### Sajith 
            
            i2t_precision_tr=i2t_corr_tr/batch_epoch_tr
            # t2i_precision_tr=t2i_corr_tr/batch_epoch_tr
            print("training precision", i2t_precision_tr)##+t2i_precision_tr)/2)
            model.eval()
            with torch.set_grad_enabled(False):

                val_loss = 0.0
                val_b=0
                total_epoch = 0
                validation_true = []
                validation_estimated = []
                n = 0
                # valid_dl.dataset.dataset.test_or_val = True
                image_embeddings_val = []
                report_embeddings_val = []
                for num_batches_valid,(images, text,labells,masks) in enumerate(valid_dl):
                    
 #               for i, batch in enumerate(valid_dl):
     #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
      #              model.to(device)
            #_ = model(batch[0][0])
    #                if torch.cuda.is_available():
                    images, labells = images.to(device), labells.to(device)
                    mask = text['attention_mask'].to(device)
                    input_id = text['input_ids'].squeeze(1).to(device)
                    
                    # pred,img_emb_q,report_emb_q,patch_emb_q,word_feat_q,word_attn_q,sents,patch_atten_output,word_atten_output = model(images.float(),input_id, mask)
                    pred,img_emb_q,report_emb_q,patch_emb_q = model(images.float(),input_id, mask)

                    image_embeddings_val.append(img_emb_q)
                    report_embeddings_val.append(report_emb_q)
                    # bz = img_emb_q.size(0)
                   
                    prob = torch.sigmoid(pred)

                    # labs = torch.arange(bz).type_as(report_emb_q).long()
                    # labels = torch.eye(bz).type_as(report_emb_q)[labs]
                    # loss_fn = ContrastiveLoss_euclidean(margin=0.1)
                    # loss0 = loss_fn(img_emb_q,report_emb_q, labels)
                    # valid_loss = loss0
                    
                    
                    for i in range(len(labells.tolist())):
                        validation_true.append(labells.tolist()[i])#[0])
                        validation_estimated.append(prob.tolist()[i])#[0])

                    if ((num_batches_valid + 1) % num_batch_accumulate == 0) or (num_batches_valid + 1 == len(valid_dl)): #If we have enough batches to take a step
                        
                        # val_batches = 0  
                        
                        val_batches+=1
                        image_embeddings_val = torch.cat(image_embeddings_val, dim=0)
                        report_embeddings_val = torch.cat(report_embeddings_val, dim=0)
                        bz = len(image_embeddings_val)
                   
                        prob = torch.sigmoid(pred)

                        labs = torch.arange(bz).type_as(report_embeddings_val).long()
                        labels = torch.eye(bz).type_as(report_embeddings_val)[labs]
                        # labels = torch.arange(bz).type_as(report_emb_q).long()
                        # euclidean_distance = torch.nn.functional.pairwise_distance(image_embeddings_val, report_embeddings_val, keepdim=True)
                        
                        # loss_fn = ContrastiveLoss_euclidean(margin=args.margin)
                        if args.similarity_measure=="euclidian":
                            loss_fn = ContrastiveLoss_euclidean(margin=args.margin)
                        else:
                            loss_fn = ContrastiveLoss_cosine2(margin=args.margin)
                        valid_loss,i_t_scores,t_i_scores = loss_fn(image_embeddings_val,report_embeddings_val,labels)
                        i_t_scores=cosine_similarity(image_embeddings_val,report_embeddings_val)
                        # print("why",image_embeddings_val.shape,report_embeddings_val.shape)
                        i2t_acc1_val,i2t_corr_val_batch,i2t_batch_val = precision_at_k(i_t_scores)#, labels, top_k=(1,))
                        # t2i_acc1_val,t2i_corr_val_batch,t2i_batch_val = precision_at_k(t_i_scores, labels, top_k=(1,))
                        i2t_corr_val+=i2t_corr_val_batch
                        batch_epoch_val+=i2t_batch_val
                        # t2i_corr_val+=t2i_corr_val_batch
                        
                        # print("validation precision",(i2t_acc1_val[0] + t2i_acc1_val[0]) / 2.)
                        # valid_loss_1 = loss_fn(report_embeddings_val,image_embeddings_val, labels)
                        # valid_loss = (valid_loss_0 + valid_loss_1)/2
                        # valid_loss = loss_fn(euclidean_distance, labels)
                        # valid_loss = loss0
                        val_loss += valid_loss.detach().item()
                        image_embeddings_val = []
                        report_embeddings_val = []
    #                corr = (pred > 0.0).squeeze().long() != labels
                    # val_err += int(corr.sum())
    #                total_epoch += len(labels)
                    n = n + 1
                val_loss = val_loss / (val_batches)#* batch_size)
                print("validation_ita",val_loss)
                i2t_precision_val=i2t_corr_val/batch_epoch_val
                # t2i_precision_val=t2i_corr_val/batch_epoch_val
                print("validation precision", i2t_precision_val)#+t2i_precision_val)/2)
                ita_list_val.append(val_loss)
                test_true = []
                test_estimated = []
                # for images, labells in test_dl:
                        
                    
                #     images, labells = images.to(device), labells.to(device)
                #     pred = model(images)
                #     prob = torch.sigmoid(pred)#F.softmax(pred, dim=1)
                #     for i in range(len(labells.tolist())):
                #         test_true.append(labells.tolist()[i][0])
                #         test_estimated.append(prob.tolist()[i][0])

                # Calculate the AUC for the different models
    #        print("sssssssssssssssssssssssss",validation_true)#len(validation_true))
            val_auc = roc_auc_score(validation_true, validation_estimated)#,multi_class='ovr')
            # test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
            train_auc = roc_auc_score(training_ture, training_estimated)#,multi_class='ovr')

                # total_val_err[epoch] = float(val_err) / total_epoch
    #            total_val_loss[epoch] = float(val_loss) / (n + 1)


        

            total_train_auc[epoch] = train_auc
            total_val_auc[epoch] = val_auc
            ##### self-added
            print("epoch",epoch,":","train_AUC:",train_auc,"val_AUC",val_auc)
            if epoch >=50 and epoch%10==0:
                print("ita_train_list",ita_list)
                print("ita_valid_list",ita_list_val)
            del valid_loss
        model.eval()
        with torch.set_grad_enabled(False):

            # val_loss = 0.0
            # val_b=0
            # total_epoch = 0
            test_true = []
            test_estimated = []
            n = 0
            # test_dl.dataset.dataset.test_or_val = True
            
            for images, text,labells,masks in test_dl:
                
#               for i, batch in enumerate(valid_dl):
    #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
    #              model.to(device)
        #_ = model(batch[0][0])
#                if torch.cuda.is_available():
                images, labells = images.to(device), labells.to(device)
                mask = text['attention_mask'].to(device)
                input_id = text['input_ids'].squeeze(1).to(device)
        
                pred,image_emb_q,report_emb_q,patch_emb_q,word_feat_q,word_attn_q,sents,patch_atten_output,word_atten_output = model(images.float(),input_id, mask)#(**texts)#

                # pred = model(images)
                # def model_wrapper(*x):
                #     return model(*x)[0]
                # print(model) 
                # print("layer",model.layer4[0].conv2)               
                # layer_gc = LayerGradCam(model, model.layer4[0].conv2) #layer4[0].conv1
                # # layer_gc.grad_cam.forward_func = model_wrapper
                # # layer_gc.guided_backprop.forward_func = model_wrapper
                # attr = layer_gc.attribute(images)
                # print(model)
                # print("attr",attr.shape)
                
                ##$$$$upsampled_attr = LayerAttribution.interpolate(attr, (240,240,155))
                # print("attr",upsampled_attr.shape)
                ####$$$$numpy_heatmap=upsampled_attr .cpu().detach().numpy()
                # print(numpy_heatmap[0][0].shape)
                # for saving:
                # for i in numpy_heatmap:
                ###$$$$ np.save(os.path.join(args.output_dir,"model_heatmap.npy"),numpy_heatmap[0])#[i][0])
        
                
                prob = torch.sigmoid(pred)#F.softmax(pred, dim=1)
                # loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                
                for i in range(len(labells.tolist())):
                    test_true.append(labells.tolist()[i])#[0])
                    test_estimated.append(prob.tolist()[i])#[0])
                
            test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
            print("test_auc",test_auc)
        if fold==0:
            break

    #        train_fpr, train_tpr, _ = roc_curve(training_true, training_estimated)
    #        val_fpr, val_tpr, _ = roc_curve(validation_true, validation_estimated)

            

    #        logging.info(
    #            "Epoch {}: Train loss: {}, Train AUC: {} | Val loss: {}, Val AUC: {} ".format(
    #                epoch + 1,
    #                total_train_loss[epoch],
    #                total_train_auc[epoch],
    #                total_val_loss[epoch],
    #                total_val_auc[epoch]))

            # model_path = get_model_name(trial=fold, name=net.name, batch_size=batch_size, learning_rate=learning_rate,
                                        # dropout_rate=0.1, epoch=epoch + 1) #after mednet commented

        # if fold==3:
        # if 
                

        # torch.save(model.state_dict(), os.path.join(args.output_dir,f"_cl_attention__fold__{fold}"))

    # np.savetxt(os.path.join(save_folder, "{}_train_err.csv".format(model_path)), total_train_err)
#    np.savetxt(os.path.join(args.output_dir, "{}_train_loss.csv".format(model_path)), total_train_loss)
#    np.savetxt(os.path.join(args.output_dir, "{}_train_auc.csv".format(model_path)), total_train_auc)
    # np.savetxt(os.path.join(save_folder, "{}_val_err.csv".format(model_path)), total_val_err)
#    np.savetxt(os.path.join(args.output_dir, "{}_val_loss.csv".format(model_path)), total_val_loss)
#    np.savetxt(os.path.join(args.output_dir, "{}_val_auc.csv".format(model_path)), total_val_auc)



    logging.info('Finished training.')
#    logging.info(f'Time elapsed: {round(time.time() - training_start_time, 3)} seconds.')

    # return total_train_err, total_train_loss, total_train_auc, total_val_err, total_val_loss, total_val_auc
    return 0, total_train_auc, 0, total_val_auc



    



def train_global_local_model(args, data_ds,test_dl, output_model_path, tuning=False):
    
    global_step = 0
    temperature=0.1
    softmax_temperature: float = 0.07
    local_temperature=0.1
   
    lambda_1: float =1
    lambda_2: float =1
    
    lambda_3: float = 0.125
    num_heads = 2
    emb_dim = 128
    
 #   optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
    batch_size=args.batch_size
    num_batch_accumulate = args.accumulate/batch_size #128 / batch_size
 
    classifier_criterion = nn.BCEWithLogitsLoss()#BCELoss()#.
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
#    optimizer.zero_grad()
#    for t in range(n_trial):
    weight_path="/hpf/largeprojects/fkhalvati/Sara/pretrain/resnet_18_23dataset.pth"
    best_auc=0

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_ds)))):
    
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        # train_dl = DataLoader(data_ds, batch_size=batch_size, sampler=train_sampler)
        valid_dl = DataLoader(data_ds, batch_size=batch_size, sampler=test_sampler)
        train_dl = DataLoader(data_ds, batch_size=batch_size, shuffle=True) #whole_data
        model=image_text_attention(emb_dim=128,num_heads=num_heads,mode="global_local")
        # print("self",model.cnn.in_planes)
        # opt={}
        ##################3pretrain
        # model = generate_model(model_depth=18, inplanes=inplanes, n_classes=1039)
        model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        # del model.cnn.local_embed
        # model.cnn.fc  = nn.Linear(512, 1)
        net_dict = model.cnn.state_dict()
        
        pretrain = torch.load(weight_path)
        pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items()}# if k in net_dict.keys()}
         
        # net_dict.update(pretrain_dict)
        model.cnn.load_state_dict(pretrain_dict)#(net_dict)
        # model.cnn.eval()
        ##############3pretrain
        # model.load_state_dict(weight_path['state_dict'])  ####mednet
        # optimizer.load_state_dict(weight_path['optimizer']) ### mednet
        # for p in parameters:
        model.cnn.self_attention = SelfAttention(512)
        model.cnn.local_embed = LocalEmbedding_3d(
            256, 256,256
        )
        # model.cnn.depth_attention = DepthAttention(512)
        # trainable_layers = []
        # for name, param in model.cnn.named_parameters():
        #     if param.requires_grad:
        #         trainable_layers.append(name)
        # print("trainable layer",trainable_layers)
        for pname, p in model.cnn.named_parameters():
            p.requires_grad = False
        for p in model.cnn.layer4.parameters():
            p.requires_grad = True
        for p in model.cnn.layer3.parameters():
            p.requires_grad = True
        for p in model.cnn.self_attention.parameters():
            p.requires_grad = True
        # for p in model.cnn.depth_attention.parameters():
        #     p.requires_grad = True
        for p in model.global_embed.parameters():
            p.requires_grad = True
        for p in model.cnn.local_embed.parameters():
            p.requires_grad = True
        # for p in model.cnn.layer2.parameters():
        #     p.requires_grad = True
        # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # patch_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # word_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # model.cnn.fc  = nn.Linear(512, 1)#nn.Sequential(nn.Linear(512, 256),nn.Linear(256, 1))#3) #512
        device=torch.device("cuda:0")
        # model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)#,device_ids = [int(i) for i in "0,1".split(',')])
        model.to(device)
        # print("gpu_id",torch.cuda.get_device_name(1),torch.cuda.get_device_name(0))
        # model.module.to(torch.device('cuda:1'))
        # patch_local_atten_layer.to(device)
        # word_local_atten_layer.to(device)

        
        component_params, total_params = count_trainable_parameters(model.cnn)
        # Print the results
        print("Components containing trainable parameters:")
        for component_name, num_params in component_params.items():
            print(f"{component_name}: {num_params} parameters")

        print(f"\nTotal number of trainable parameters: {total_params}")


        optimizer = optim.AdamW(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=400,T_mult=1, eta_min=1e-8, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        #(optimizer, step_size=10, gamma=0.1)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        optimizer.zero_grad()
        ita_list=[]
        ita_list_val=[]
        local_list=[]
        cl_list=[]
        
        loss_fn = DynamicWeightedLoss(num_losses=2)
        for epoch in range(args.num_epochs):
            model.train()
            
            # model.cnn.eval()
            # model.cnn.layer4.train()
            # model.cnn.layer3.train()
            # model.cnn.layer4.train()
            train_loss = 0
            counter = 0
            num_batches=0
            num_batches_valid=0
            val_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]
     #       print(len(train_dl))
            
            #******
            # image_embeddings = []
            
            report_embeddings = []
            image_embeddings=[]
            label_list=[]
            pred_list=[]
            word_embeddings=[]
            word_attention=[]
            weighted_image_context=[]
            patch_embeddings=[]
            sent_list=[]
            patch_attention_list=[]
            patch_weights=[]
            word_weights=[]
            merged_att_list = []
            cap_lens_list=[]

            i2t_corr_tr=0
            batch_epoch_tr=0
            i2t_corr_val=0
            t2i_corr_tr=0
            t2i_corr_val=0
            batch_epoch_val=0
            precision_list=[]
            loc_list = []
            # image_embeddings_1 = torch.Tensor().cuda()
            # image_embeddings_2= torch.Tensor().cuda()
            # image_embeddings = torch.Tensor().cuda()
            # report_embeddings = torch.Tensor().cuda()
            print("epoch",epoch)
            for num_batches,(images,text, labells,masks,tumor_location) in enumerate(train_dl): 
                # print("HI")
                # print("report",len(report_embeddings))
                device="cuda:0"#torch.device("cuda:0")
                images= images.to(device)#,labells.to(device)
                
                mask = text['attention_mask'].to(device)
                input_id = text['input_ids'].squeeze(1).to(device)

 
    #            loss_classifier = classifier_criterion(y_pred, labels)

    #            epoch_loss += total_loss.item()

                
                # model.to(device)
                # optimizer.zero_grad()
                # pred=model(images)
                images=images.float()
                # pred,img_emb_q,report_emb_q,patch_emb_q,word_emb_q,word_attn_q,sents,patch_atten_output,word_atten_output= model(images,input_id, mask)
                img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_atten_output_weight,word_atten_output_weight,cap_lens,patch_atten_output,word_atten_output,merged_att= model(images,input_id, mask)
                # print("print output",word_atten_output.shape,word_atten_output_weight.shape)
                # print("patch_emb_q",patch_emb_q.shape)
                # print("word_emb_q.shape",word_emb_q.shape)
                
                
                # print("params",model.parameters)
                # del patch_emb_q,word_emb_q,word_attn_q,sents,patch_atten_output,word_atten_output
                # pred=pred.detach()
                # prob = torch.sigmoid(pred)

                # attnT = torch.transpose(word_atten_output, 1, 2).contiguous()

    
                # weightedContext = torch.bmm(patch_emb_q, attnT)
                image_embeddings.append(img_emb_q)
                # image_embeddings = np.append(image_embeddings, img_emb_q)
                report_embeddings.append(report_emb_q)
                word_attention.append(word_atten_output)
                word_embeddings.append(word_emb_q)
                patch_embeddings.append(patch_emb_q)
                sent_list.append(sents)
                patch_attention_list.append(patch_atten_output)
                cap_lens_list.append(cap_lens)
                merged_att_list.append(merged_att)
                loc_list.append(tumor_location)
                # patch_weights.append(patch_atten_output_weight)
                # word_weights.append(word_atten_output_weight)
                
                # euclidean_distance = torch.nn.functional.pairwise_distance(img_emb_q, report_emb_q, keepdim=True)
                # # torch.cuda.empty_cache()
                # # distance_list.append(euclidean_distance)
                # img_emb_q = img_emb_q.unsqueeze(1)
                # report_emb_q = report_emb_q.unsqueeze(1)
                # del pred,prob
                # if len(report_embeddings)<=4:

                #     image_embeddings_1 = torch.cat((image_embeddings_1, img_emb_q))#, dim=1)
                # else:
                #     image_embeddings_2 = torch.cat((image_embeddings_2, img_emb_q))
                # image_embeddings = torch.cat((image_embeddings, img_emb_q),dim=1)

                # # print("image",len(image_embeddings_1),len(image_embeddings_2))
                # report_embeddings = torch.cat((report_embeddings, report_emb_q), dim=1)
                # print("lennn",len(report_embeddings),len(image_embeddings))
                # print("embb",report_emb_q.shape)
                if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                    #  optimizer.zero_grad()
                    # print("Hi")
                    # distance_list = torch.cat(distance_list, dim=0)
                    # print("num_batches",num_batches)
                    # print("outer:",cap_lens_list)
                    cap_len_list = [element for sublist in cap_lens_list for element in sublist]

                    report_embeddings = torch.cat(report_embeddings, dim=0)
                    # print("reporttttt",report_embeddings.shape)
                    image_embeddings=torch.cat(image_embeddings, dim=0)   
                    merged_att_list=torch.cat(merged_att_list,dim=0)
                    word_attention=torch.cat(word_attention, dim=0) 
                    word_embeddings=torch.cat(word_embeddings, dim=0) 
                    patch_embeddings=torch.cat(patch_embeddings, dim=0) 
                    # sent_list=torch.cat(sent_list, dim=0) 
                    patch_attention_list=torch.cat(patch_attention_list, dim=0) 
                    # patch_weights=torch.cat(patch_weights, dim=0) 
                    # sent_list=torch.cat(sent_list, dim=0) 
                    # word_weights=torch.cat(word_weights, dim=0) 
                    # cap_lens_list=torch.cat(cap_lens_list, dim=0) 
                    # loss_classification=classifier_criterion(pred_batch,label_batch.unsqueeze(1).float())   
                    # print("imageee",image_embeddings.shape)   
                    loc_list=torch.cat(loc_list, dim=0)                
                    bz = len(report_embeddings)
                    
                    # distance_list=torch.tensor(distance_list, requires_grad=True)
                    # distance_list=distance_list.cuda()
                    labs = torch.arange(bz).type_as(report_emb_q).long()
                    labels = torch.eye(bz).type_as(report_emb_q)[labs]
                    # print("labellllls",labels.shape)
                    if args.similarity_measure=="euclidian":
                        # print("ok")
                        loss_g = ContrastiveLoss_euclidean(margin=args.margin)
                    else:
                        loss_g = ContrastiveLoss_cosine2(margin=args.margin,mode="global")
                    
                    
                    loss_global,_,t_i_scores = loss_g(image_embeddings,report_embeddings, labels,loc_list)#+loss_classification  #(distance_list,labels)
                    # print("scores",i_t_scores.shape)
                    # i2t_acc1_tr = precision_at_k(i_t_scores, labels, top_k=(1,))
                    # t2i_acc1_tr = precision_at_k(t_i_scores, labels, top_k=(1,))
                    # print("training precision",(i2t_acc1_tr + t2i_acc1_tr) / 2.)
                    # print("naaannnn",loc_list)
                    # print("naaannnn2",image_embeddings)
                    # print("naaannnn3",report_embeddings)
                    # print("naaannnn4",labels)
                    
                    loss_local_0 = local_contrastive_loss2( patch_embeddings,word_embeddings,word_attention, cap_len_list,loc_list,margin=args.margin)
                    
                    loss_local_1 = local_contrastive_loss2( word_embeddings,patch_embeddings,patch_attention_list, cap_len_list,loc_list,margin=args.margin)
                    
                    # loss_1 = loss_fn(report_embeddings,image_embeddings, labels) 
                    # loss0 = (loss_0+loss_1)/2
                    
                    # optimizer.zero_grad()

                    ### local loss
                    loss_local=loss_local_0+loss_local_1
                    # print("heyy",loss_global,loss_local)
                    # print("loss hereeee",loss_global.view(1),loss_global)
                    total_train_loss_stack=torch.cat([loss_global.view(1), loss_local.view(1)])
                    # loss0,loss_weights=loss_fn(total_train_loss_stack)
                    # if epoch<=30:
                    #     loss0 = 0.1*loss_global + loss_local
                    # else:
                    #     loss0 = loss_global + 0.1*loss_local
                    # loss0 = loss_local + loss_global
                    loss0 = loss_local
                    i_t_scores=cosine_similarity(image_embeddings,report_embeddings)
                    i2t_acc1_tr,i2t_corr_tr_batch,i2t_batch_tr = precision_at_k(i_t_scores)#, labels, top_k=(1,))
                    # t2i_acc1_tr,t2i_corr_tr_batch,t2i_batch_tr  = precision_at_k(t_i_scores)#, labels, top_k=(1,))
                    print("pr",i2t_acc1_tr)
                    i2t_corr_tr+=i2t_corr_tr_batch
                    # t2i_corr_tr+=t2i_corr_tr_batch
                    
                    batch_epoch_tr+=i2t_batch_tr

                    loss0.backward()


                    optimizer.step() 
                    # for element in distance_list:
                    #     # print(element.requires_grad)
                    #     print("element",element.grad)
                    optimizer.zero_grad() 
                    train_loss+=loss0.detach().item()#*batch_size ### Sajith: added detach 
                    train_batches+=1

                    # distance_list = []
                    image_embeddings = []
                    report_embeddings = []
                    word_embeddings=[]
                    word_attention=[]
                    
                    loc_list = []
                    word_attention=[]
                    
                    patch_embeddings=[]
                    sent_list=[]
                    patch_attention_list=[]
                    cap_lens_list=[]
                    word_weights=[]
                    patch_weights=[]
                    merged_att_list=[]
                    # image_embeddings_1 = torch.Tensor().cuda()
                    # image_embeddings = torch.Tensor().cuda()

                    # report_embeddings = torch.Tensor().cuda()
                # for i in range(len(labells.tolist())):
                #     # print("ll",labells)
                    
                #     training_ture.append(labells.tolist()[i])#[0])
                #     training_estimated.append(prob.tolist()[i])#[0]) #prob
               #     model_grad = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)

    #            corr = (pred>0.0).squeeze().long() != labells
                # train_loss+=loss.detach().item()*batch_size ### Sajith: added detach 
                #?
                # train_batches+=1
                counter += 1

            # Calculate average over epoch
            # total_train_err[epoch] = float(train_err) / total_epoch
    ##        total_train_loss[epoch] = float(train_loss) / (n + 1)
            # scheduler.step()
            train_loss = train_loss/(train_batches)#*batch_size)  
            print("loss_ita",train_loss)
            ita_list.append(train_loss)
            # scheduler.step()
            # global only: _if_updated_location_attention_local_global__fold
            if epoch%10==0 and epoch>330:
                torch.save(model.state_dict(), os.path.join(args.output_dir,f"_local_if_updated_location_attention_local_global__fold__{fold}__epoch__{epoch}__margin{args.margin}"))
            del loss0 ### Sajith 
            i2t_precision_tr=i2t_corr_tr/batch_epoch_tr
            # t2i_precision_tr=t2i_corr_tr/batch_epoch_tr
            print("training precision", (i2t_precision_tr))#+t2i_precision_tr)/2)
            precision_list.append(i2t_precision_tr)
            model.eval()
            with torch.set_grad_enabled(False):

                val_loss = 0.0
                val_b=0
                total_epoch = 0
                validation_true = []
                validation_estimated = []
                n = 0
                # valid_dl.dataset.dataset.test_or_val = True
                image_embeddings_val = []
                report_embeddings_val = []
    
                word_attention_val=[]
                word_embeddings_val=[]
                patch_embeddings_val=[]
                sent_list_val=[]
                patch_attention_list_val=[]
                cap_lens_list_val=[]
                patch_weights_val=[]
                word_weights_val=[]
                merged_att_list_val=[]
                val_loc_list = []
                for num_batches_valid,(images, text,labells,masks,tumor_location) in enumerate(valid_dl):
                    
 #               for i, batch in enumerate(valid_dl):
     #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
      #              model.to(device)
            #_ = model(batch[0][0])
    #                if torch.cuda.is_available():
                    images = images.to(device)
                    mask = text['attention_mask'].to(device)
                    input_id = text['input_ids'].squeeze(1).to(device)
                    
                    # pred,img_emb_q,report_emb_q,patch_emb_q,word_feat_q,word_attn_q,sents,patch_atten_output,word_atten_output = model(images.float(),input_id, mask)
                    img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_atten_output_weight,word_atten_output_weight,cap_lens,patch_atten_output,word_atten_output,merged_att = model(images.float(),input_id, mask)

                    image_embeddings_val.append(img_emb_q)
                    report_embeddings_val.append(report_emb_q)
                    word_attention_val.append(word_atten_output)
                    word_embeddings_val.append(word_emb_q)
                    patch_embeddings_val.append(patch_emb_q)
                    sent_list_val.append(sents)
                    patch_attention_list_val.append(patch_atten_output)
                    cap_lens_list_val.append(cap_lens)
                    patch_weights_val.append(patch_atten_output_weight)
                    word_weights_val.append(word_atten_output_weight)
                    merged_att_list_val.append(merged_att)
                    val_loc_list.append(tumor_location)
                    # bz = img_emb_q.size(0)
                   
                    # prob = torch.sigmoid(pred)

                    # labs = torch.arange(bz).type_as(report_emb_q).long()
                    # labels = torch.eye(bz).type_as(report_emb_q)[labs]
                    # loss_fn = ContrastiveLoss_euclidean(margin=0.1)
                    # loss0 = loss_fn(img_emb_q,report_emb_q, labels)
                    # valid_loss = loss0
                    
                    
                    # for i in range(len(labells.tolist())):
                    #     validation_true.append(labells.tolist()[i])#[0])
                    #     validation_estimated.append(prob.tolist()[i])#[0])

                    if ((num_batches_valid + 1) % num_batch_accumulate == 0) or (num_batches_valid + 1 == len(valid_dl)): #If we have enough batches to take a step
                        
                        # val_batches = 0  
                        cap_len_list_val = [element for sublist in cap_lens_list_val for element in sublist]
                        val_batches+=1
                        image_embeddings_val = torch.cat(image_embeddings_val, dim=0)
                        report_embeddings_val = torch.cat(report_embeddings_val, dim=0)
                        word_attention_val=torch.cat(word_attention_val, dim=0)
                        word_embeddings_val=torch.cat(word_embeddings_val, dim=0)
                        patch_embeddings_val=torch.cat(patch_embeddings_val, dim=0)
                        # sent_list_val=torch.cat(sent_list_val, dim=0)
                        patch_attention_list_val=torch.cat(patch_attention_list_val, dim=0)
                        patch_weights_val=torch.cat(patch_weights_val, dim=0)
                        # sent_list_val=torch.cat(sent_list_val, dim=0)
                        word_weights_val=torch.cat(word_weights_val, dim=0)
                        merged_att_list_val=torch.cat(merged_att_list_val,dim=0)
                        val_loc_list=torch.cat(val_loc_list,dim=0)
                        bz = len(image_embeddings_val)
                   
                        # prob = torch.sigmoid(pred)

                        labs = torch.arange(bz).type_as(report_embeddings_val).long()
                        labels = torch.eye(bz).type_as(report_embeddings_val)[labs]

                        # euclidean_distance = torch.nn.functional.pairwise_distance(image_embeddings_val, report_embeddings_val, keepdim=True)
                        
                        # loss_fn = ContrastiveLoss_euclidean(margin=args.margin)
                        if args.similarity_measure=="euclidian":
                            valid_loss_g = ContrastiveLoss_euclidean(margin=args.margin)
                        else:
                            valid_loss_g = ContrastiveLoss_cosine2(margin=args.margin,mode="global")
                        
                        

                        valid_loss_global,i_t_scores,t_i_scores = valid_loss_g(image_embeddings_val,report_embeddings_val, labels,val_loc_list)
                        # i2t_acc1_val = precision_at_k(i_t_scores, labels, top_k=(1,))
                        # t2i_acc1_val = precision_at_k(t_i_scores, labels, top_k=(1,))
                        # print("validation precision",(i2t_acc1_val + t2i_acc1_val) / 2.)

                        valid_loss_local_0 = local_contrastive_loss2(word_embeddings_val, patch_embeddings_val,patch_attention_list_val, cap_len_list_val,val_loc_list,margin=args.margin)
                        valid_loss_local_1 = local_contrastive_loss2(patch_embeddings_val, word_embeddings_val,word_attention_val, cap_len_list_val,val_loc_list,margin=args.margin)
                        # valid_loss_1 = loss_fn(report_embeddings_val,image_embeddings_val, labels)
                        # valid_loss = (valid_loss_0 + valid_loss_1)/2
                        # valid_loss = loss_fn(euclidean_distance, labels)
                        # valid_loss = loss0
                        valid_loss_local=valid_loss_local_0+valid_loss_local_1
                        valid_losses = torch.cat([valid_loss_global.view(1), valid_loss_local.view(1)])

                        # weighted_losses = valid_losses * loss_weights.unsqueeze(0).cuda()
                        # valid_loss = torch.sum(weighted_losses)
                        valid_loss = valid_loss_local + valid_loss_global
                        i2t_acc1_val,i2t_corr_val_batch,i2t_batch_val = precision_at_k(i_t_scores)#, labels, top_k=(1,))
                        # t2i_acc1_val,t2i_corr_val_batch,t2i_batch_val = precision_at_k(t_i_scores)#, labels, top_k=(1,))
                        i2t_corr_val+=i2t_corr_val_batch
                        batch_epoch_val+=i2t_batch_val
                        # t2i_corr_val+=t2i_corr_val_batch

                        val_loss += valid_loss.detach().item()
                        image_embeddings_val=[]
                        report_embeddings_val=[]
                        word_attention_val=[]
                        word_embeddings_val=[]
                        patch_embeddings_val=[]
                        sent_list_val=[]
                        patch_attention_list_val=[]
                        cap_lens_list_val=[]
                        patch_weights_val=[]
                        word_weights_val=[]
                        merged_att_list_val=[]
                        val_loc_list = []
    #                corr = (pred > 0.0).squeeze().long() != labels
                    # val_err += int(corr.sum())
    #                total_epoch += len(labels)
                    n = n + 1
                val_loss = val_loss / (val_batches)#* batch_size)
                print("validation_ita",val_loss)

                i2t_precision_val=i2t_corr_val/batch_epoch_val
                # t2i_precision_val=t2i_corr_val/batch_epoch_val
                print("validation precision", (i2t_precision_val))#+t2i_precision_val)/2)
                
                ita_list_val.append(val_loss)
                test_true = []
                test_estimated = []
                # for images, labells in test_dl:
                        
                    
                #     images, labells = images.to(device), labells.to(device)
                #     pred = model(images)
                #     prob = torch.sigmoid(pred)#F.softmax(pred, dim=1)
                #     for i in range(len(labells.tolist())):
                #         test_true.append(labells.tolist()[i][0])
                #         test_estimated.append(prob.tolist()[i][0])

                # Calculate the AUC for the different models
    #        print("sssssssssssssssssssssssss",validation_true)#len(validation_true))
            # val_auc = roc_auc_score(validation_true, validation_estimated)#,multi_class='ovr')
            # # test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
            # train_auc = roc_auc_score(training_ture, training_estimated)#,multi_class='ovr')

                # total_val_err[epoch] = float(val_err) / total_epoch
    #            total_val_loss[epoch] = float(val_loss) / (n + 1)


        

            
            ##### self-added
            
            if epoch >=50 and epoch%10==0:
                print("ita_train_list",ita_list)
                print("ita_valid_list",ita_list_val)
                print("precision_list",precision_list)
            del valid_loss
        model.eval()
        with torch.set_grad_enabled(False):

            # val_loss = 0.0
            # val_b=0
            # total_epoch = 0
            test_true = []
            test_estimated = []
            n = 0
            # test_dl.dataset.dataset.test_or_val = True
            
            for images, text,labells,masks,tumor_location in test_dl:
                
#               for i, batch in enumerate(valid_dl):
    #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
    #              model.to(device)
        #_ = model(batch[0][0])
#                if torch.cuda.is_available():
                images= images.to(device)
                mask = text['attention_mask'].to(device)
                input_id = text['input_ids'].squeeze(1).to(device)
        
                img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_atten_output_weight,word_atten_output_weight,cap_lens,patch_atten_output,word_atten_output,merged_att = model(images.float(),input_id, mask)#(**texts)#
               
                # pred = model(images)
                # def model_wrapper(*x):
                #     return model(*x)[0]
                # print(model) 
                # print("layer",model.layer4[0].conv2)               
                # layer_gc = LayerGradCam(model, model.layer4[0].conv2) #layer4[0].conv1
                # # layer_gc.grad_cam.forward_func = model_wrapper
                # # layer_gc.guided_backprop.forward_func = model_wrapper
                # attr = layer_gc.attribute(images)
                # print(model)
                # print("attr",attr.shape)
                
                ##$$$$upsampled_attr = LayerAttribution.interpolate(attr, (240,240,155))
                # print("attr",upsampled_attr.shape)
                ####$$$$numpy_heatmap=upsampled_attr .cpu().detach().numpy()
                # print(numpy_heatmap[0][0].shape)
                # for saving:
                # for i in numpy_heatmap:
                ###$$$$ np.save(os.path.join(args.output_dir,"model_heatmap.npy"),numpy_heatmap[0])#[i][0])
        
                
                #F.softmax(pred, dim=1)
                # loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                
                
                
            
        if fold==0:
            break

    #        train_fpr, train_tpr, _ = roc_curve(training_true, training_estimated)
    #        val_fpr, val_tpr, _ = roc_curve(validation_true, validation_estimated)

            

    #        logging.info(
    #            "Epoch {}: Train loss: {}, Train AUC: {} | Val loss: {}, Val AUC: {} ".format(
    #                epoch + 1,
    #                total_train_loss[epoch],
    #                total_train_auc[epoch],
    #                total_val_loss[epoch],
    #                total_val_auc[epoch]))

            # model_path = get_model_name(trial=fold, name=net.name, batch_size=batch_size, learning_rate=learning_rate,
                                        # dropout_rate=0.1, epoch=epoch + 1) #after mednet commented

        # if fold==3:
        # if 
                

        # torch.save(model.state_dict(), os.path.join(args.output_dir,f"_cl_attention__fold__{fold}"))

    # np.savetxt(os.path.join(save_folder, "{}_train_err.csv".format(model_path)), total_train_err)
#    np.savetxt(os.path.join(args.output_dir, "{}_train_loss.csv".format(model_path)), total_train_loss)
#    np.savetxt(os.path.join(args.output_dir, "{}_train_auc.csv".format(model_path)), total_train_auc)
    # np.savetxt(os.path.join(save_folder, "{}_val_err.csv".format(model_path)), total_val_err)
#    np.savetxt(os.path.join(args.output_dir, "{}_val_loss.csv".format(model_path)), total_val_loss)
#    np.savetxt(os.path.join(args.output_dir, "{}_val_auc.csv".format(model_path)), total_val_auc)



    logging.info('Finished training.')
#    logging.info(f'Time elapsed: {round(time.time() - training_start_time, 3)} seconds.')

    # return total_train_err, total_train_loss, total_train_auc, total_val_err, total_val_loss, total_val_auc
    return 0, total_train_auc, 0, total_val_auc
            

def train_local_model(args, data_ds,test_dl, output_model_path, tuning=False):
    
    global_step = 0
    temperature=0.1
    softmax_temperature: float = 0.07
    local_temperature=0.1
   
    lambda_1: float =1
    lambda_2: float =1
    
    lambda_3: float = 0.125
    num_heads = 1
    emb_dim = 128
    
 #   optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
    batch_size=args.batch_size
    num_batch_accumulate = args.accumulate/batch_size #128 / batch_size
 
    classifier_criterion = nn.BCEWithLogitsLoss()#BCELoss()#.
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
#    optimizer.zero_grad()
#    for t in range(n_trial):
    weight_path="/hpf/largeprojects/fkhalvati/Sara/pretrain/resnet_18_23dataset.pth"
    best_auc=0
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_ds)))):
    
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        # train_dl = DataLoader(data_ds, batch_size=batch_size, sampler=train_sampler)
        valid_dl = DataLoader(data_ds, batch_size=batch_size, sampler=test_sampler)
        train_dl = DataLoader(whole_data, batch_size=batch_size, shuffle=True)
        model=image_text_attention(emb_dim=128,num_heads=num_heads,mode="global_local")
        # print("self",model.cnn.in_planes)
        # opt={}
        ##################3pretrain
        # model = generate_model(model_depth=18, inplanes=inplanes, n_classes=1039)
        model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        # model.cnn.fc  = nn.Linear(512, 1)
        net_dict = model.cnn.state_dict()
        
        pretrain = torch.load(weight_path)
        pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items()}# if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.cnn.load_state_dict(net_dict)
        # model.cnn.eval()
        ##############3pretrain
        # model.load_state_dict(weight_path['state_dict'])  ####mednet
        # optimizer.load_state_dict(weight_path['optimizer']) ### mednet
        # for p in parameters:
        model.cnn.self_attention = SelfAttention(512)
        model.cnn.local_embed = LocalEmbedding_3d(
            256, 256,256
        )
        for pname, p in model.cnn.named_parameters():
            p.requires_grad = False
        for p in model.cnn.layer4.parameters():
            p.requires_grad = True
        for p in model.cnn.layer3.parameters():
            p.requires_grad = True
        for p in model.cnn.self_attention.parameters():
            p.requires_grad = True


        for p in model.cnn.local_embed.parameters():
            p.requires_grad = True
        # for p in model.patch_local_atten_layer.parameters():
        #     p.requires_grad = True
        # for p in model.word_local_atten_layer.parameters():
        #     p.requires_grad = True
        # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # patch_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # word_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # model.cnn.fc  = nn.Linear(512, 1)#nn.Sequential(nn.Linear(512, 256),nn.Linear(256, 1))#3) #512
        device=torch.device("cuda:0")
        # model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)#,device_ids = [int(i) for i in "0,1".split(',')])
        model.to(device)
        # print("gpu_id",torch.cuda.get_device_name(1),torch.cuda.get_device_name(0))
        # model.module.to(torch.device('cuda:1'))
        # patch_local_atten_layer.to(device)
        # word_local_atten_layer.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=400,T_mult=1, eta_min=1e-8, last_epoch=-1)
            
        #(optimizer, step_size=10, gamma=0.1)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        optimizer.zero_grad()
        ita_list=[]
        ita_list_val=[]
        local_list=[]
        cl_list=[]
        for epoch in range(args.num_epochs):
            model.train()
            # model.transformer.train()
            # # model.cnn.eval()
            # model.cnn.layer4.train()
            train_loss = 0
            counter = 0
            num_batches=0
            num_batches_valid=0
            val_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]
            i2t_corr_tr=0
            batch_epoch_tr=0
            i2t_corr_val=0
            t2i_corr_tr=0
            t2i_corr_val=0
            batch_epoch_val=0
     #       print(len(train_dl))
            
            #******
            # image_embeddings = []
            
            report_embeddings = []
            image_embeddings=[]
            label_list=[]
            pred_list=[]
            word_embeddings=[]
            word_attention=[]
            weighted_image_context=[]
            patch_embeddings=[]
            sent_list=[]
            patch_attention_list=[]
            patch_weights=[]
            word_weights=[]
            merged_att_list = []
            cap_lens_list=[]
            loc_list = []
            # image_embeddings_1 = torch.Tensor().cuda()
            # image_embeddings_2= torch.Tensor().cuda()
            # image_embeddings = torch.Tensor().cuda()
            # report_embeddings = torch.Tensor().cuda()
            
            for num_batches,(images,text, labells,masks,location) in enumerate(train_dl): 
                # print("HI")
                # print("report",len(report_embeddings))
                device="cuda:0"#torch.device("cuda:0")
                images= images.to(device)#,labells.to(device)
                # labells= labells.to(device)
                mask = text['attention_mask'].to(device)
                input_id = text['input_ids'].squeeze(1).to(device)

 
    #            loss_classifier = classifier_criterion(y_pred, labels)

    #            epoch_loss += total_loss.item()

                
                # model.to(device)
                # optimizer.zero_grad()
                # pred=model(images)
                images=images.float()
                # pred,img_emb_q,report_emb_q,patch_emb_q,word_emb_q,word_attn_q,sents,patch_atten_output,word_atten_output= model(images,input_id, mask)
                img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_atten_output_weight,word_atten_output_weight,cap_lens,patch_atten_output,word_atten_output,merged_att= model(images,input_id, mask)
                # print("trainbale",model.patch_local_atten_layer.gamma.requires_grad,model.word_local_atten_layer.gamma.requires_grad)

                # for p in model.patch_local_atten_layer.parameters():
                #     p.requires_grad = True
                # for p in model.word_local_atten_layer.parameters():
                #     p.requires_grad = True
                # print("print output",word_atten_output.shape,word_atten_output_weight.shape)
                # print("patch_emb_q",patch_emb_q.shape)
                # pred_list.append(pred)
                # label_list.append(labells)
                # print("params",model.parameters)
                # del patch_emb_q,word_emb_q,word_attn_q,sents,patch_atten_output,word_atten_output
                # pred=pred.detach()
                # prob = torch.sigmoid(pred)

                # attnT = torch.transpose(word_atten_output, 1, 2).contiguous()

    
                # weightedContext = torch.bmm(patch_emb_q, attnT)
                merged_att_list.append(merged_att.detach())
                image_embeddings.append(img_emb_q)
                # image_embeddings = np.append(image_embeddings, img_emb_q)
                report_embeddings.append(report_emb_q)
                word_attention.append(word_atten_output)
                word_embeddings.append(word_emb_q)
                patch_embeddings.append(patch_emb_q)
                sent_list.append(sents)
                patch_attention_list.append(patch_atten_output)
                cap_lens_list.append(cap_lens)
                patch_weights.append(patch_atten_output_weight)
                word_weights.append(word_atten_output_weight)
                loc_list.append(location)
                
                # euclidean_distance = torch.nn.functional.pairwise_distance(img_emb_q, report_emb_q, keepdim=True)
                # # torch.cuda.empty_cache()
                # # distance_list.append(euclidean_distance)
                # img_emb_q = img_emb_q.unsqueeze(1)
                # report_emb_q = report_emb_q.unsqueeze(1)
                # del pred,prob
                # if len(report_embeddings)<=4:

                #     image_embeddings_1 = torch.cat((image_embeddings_1, img_emb_q))#, dim=1)
                # else:
                #     image_embeddings_2 = torch.cat((image_embeddings_2, img_emb_q))
                # image_embeddings = torch.cat((image_embeddings, img_emb_q),dim=1)

                # # print("image",len(image_embeddings_1),len(image_embeddings_2))
                # report_embeddings = torch.cat((report_embeddings, report_emb_q), dim=1)
                # print("lennn",len(report_embeddings),len(image_embeddings))
                # print("embb",report_emb_q.shape)
                if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                    #  optimizer.zero_grad()
                    # print("Hi")
                    # distance_list = torch.cat(distance_list, dim=0)
                    # print("num_batches",num_batches)
                    # print("outer:",cap_lens_list)
                    cap_len_list = [element for sublist in cap_lens_list for element in sublist]

                    report_embeddings = torch.cat(report_embeddings, dim=0)
                    # print("reporttttt",report_embeddings.shape)
                    image_embeddings=torch.cat(image_embeddings, dim=0)   
                    # pred_batch=torch.cat(pred_list, dim=0) 
                    # label_batch=torch.cat(label_list, dim=0) 

                    merged_att_list=torch.cat(merged_att_list, dim=0)   

                    word_attention=torch.cat(word_attention, dim=0) 
                    word_embeddings=torch.cat(word_embeddings, dim=0) 
                    patch_embeddings=torch.cat(patch_embeddings, dim=0) 
                    # sent_list=torch.cat(sent_list, dim=0) 
                    patch_attention_list=torch.cat(patch_attention_list, dim=0) 
                    patch_weights=torch.cat(patch_weights, dim=0) 
                    # sent_list=torch.cat(sent_list, dim=0) 
                    word_weights=torch.cat(word_weights, dim=0) 
                    loc_list =torch.cat(loc_list, dim=0) 
                    # cap_lens_list=torch.cat(cap_lens_list, dim=0) 
                    # loss_classification=classifier_criterion(pred_batch,label_batch.unsqueeze(1).float())   
                    # print("imageee",image_embeddings.shape)                  
                    bz = len(report_embeddings)
                    
                    # distance_list=torch.tensor(distance_list, requires_grad=True)
                    # distance_list=distance_list.cuda()
                    labs = torch.arange(bz).type_as(report_emb_q).long()
                    labels = torch.eye(bz).type_as(report_emb_q)[labs]
                    # print("labellllls",labels.shape)
                    # if args.similarity_measure=="euclidian":
                    #     # print("ok")
                    #     loss_g = ContrastiveLoss_euclidean(margin=args.margin)
                    # else:
                    #     loss_g,i_t_scores,t_i_scores = ContrastiveLoss_cosine(margin=args.margin)
                    # loss_global = loss_g(image_embeddings,report_embeddings, labels)#+loss_classification  #(distance_list,labels)


                    # with torch.no_grad():
                    #     atten_weights = merged_att_list.detach()
                        
                    #     word_atten_weights = []
                    #     for i in range(bz):
                            
                    #         atten_weight = atten_weights[i]
                    #         print
                    #         nonzero = atten_weight.nonzero().squeeze()
                    #         low = torch.quantile(atten_weight[nonzero], 0.1)
                    #         high = torch.quantile(atten_weight[nonzero], 0.9)
                    #         atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                    #         word_atten_weights.append(atten_weight.clone())
                    #     word_atten_weights = torch.stack(word_atten_weights)
                    #     # TODO: maybe clip the tensor of 10 percentile and 90 percentile

                    # word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)


                    # loss_local_0 = local_contrastive_loss3( patch_embeddings,word_embeddings,word_attention, cap_len_list,word_atten_weights,margin=args.margin)
                    # # loss_local_0 = local_loss( patch_embeddings,word_embeddings, cap_len_list)
                    # loss_local_1 = local_contrastive_loss3( word_embeddings,patch_embeddings,patch_attention_list, cap_len_list,word_atten_weights,margin=args.margin)
                    # # loss_local_1 = local_loss( word_embeddings,patch_embeddings, cap_len_list)
                    # # loss_1 = loss_fn(report_embeddings,image_embeddings, labels) 
                    # # loss0 = (loss_0+loss_1)/2
                    loss_local_0 = local_contrastive_loss2( patch_embeddings,word_embeddings,word_attention, cap_len_list,loc_list,margin=args.margin) #local_loss(patch_embeddings, word_embeddings, cap_len_list,loc_list)
                    # loss_local_0 = local_loss( patch_embeddings,word_embeddings, cap_len_list)
                    loss_local_1 = local_contrastive_loss2( patch_embeddings,patch_embeddings,patch_attention_list, cap_len_list,loc_list,margin=args.margin)
                    # optimizer.zero_grad()

                    ### local loss
                    loss0=loss_local_0+loss_local_1#loss_fn(loss_global,loss_local)
                    
                    # i_t_scores = cosine_similarity2(word_embeddings,word_attention)#(image_embeddings,report_embeddings)
                    # i_t_scores = cosine_distance(word_embeddings,word_attention)[1]
                    t_i_scores = F.cosine_similarity(word_embeddings.reshape(bz,-1).unsqueeze(1), word_attention.reshape(bz,-1).unsqueeze(0), dim=-1)
                    i_t_scores = F.cosine_similarity(patch_embeddings.reshape(bz,-1).unsqueeze(1), patch_attention_list.reshape(bz,-1).unsqueeze(0), dim=-1)
                    i2t_acc1_tr,i2t_corr_tr_batch,i2t_batch_tr = precision_at_k(i_t_scores)#, labels, top_k=(1,))
                    # t_i_scores = cosine_similarity2(patch_embeddings,patch_attention_list)#(image_embeddings,report_embeddings)
                    # t_i_scores = cosine_distance(patch_embeddings,patch_attention_list)[1]
                    t2i_acc1_tr,t2i_corr_tr_batch,t2i_batch_tr = precision_at_k(t_i_scores)#, labels, top_k=(1,))
                    print("pr1",i2t_acc1_tr)
                    print("pr2",t2i_acc1_tr)
                    # t2i_acc1_tr,t2i_corr_tr_batch,t2i_batch_tr  = precision_at_k(t_i_scores)#, labels, top_k=(1,))
                    # print("test prec",i2t_acc1_tr,t2i_acc1_tr)
                    i2t_corr_tr+=i2t_corr_tr_batch
                    t2i_corr_tr+=t2i_corr_tr_batch
                    
                    batch_epoch_tr+=i2t_batch_tr
                    loss0.backward()


                    optimizer.step() 
                    # for element in distance_list:
                    #     # print(element.requires_grad)
                    #     print("element",element.grad)
                    optimizer.zero_grad() 
                    train_loss+=loss0.detach().item()#*batch_size ### Sajith: added detach 
                    train_batches+=1

                    # distance_list = []
                    image_embeddings = []
                    report_embeddings = []
                    word_embeddings=[]
                    word_attention=[]
                    
        
                    word_attention=[]
                    
                    patch_embeddings=[]
                    sent_list=[]
                    patch_attention_list=[]
                    cap_lens_list=[]
                    word_weights=[]
                    patch_weights=[]
                    merged_att_list = []
                    loc_list = []
                    # image_embeddings_1 = torch.Tensor().cuda()
                    # image_embeddings = torch.Tensor().cuda()

                    # report_embeddings = torch.Tensor().cuda()
                # for i in range(len(labells.tolist())):
                #     # print("ll",labells)
                    
                #     training_ture.append(labells.tolist()[i])#[0])
                #     training_estimated.append(prob.tolist()[i])#[0]) #prob
               #     model_grad = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)

    #            corr = (pred>0.0).squeeze().long() != labells
                # train_loss+=loss.detach().item()*batch_size ### Sajith: added detach 
                #?
                # train_batches+=1
                counter += 1

            # Calculate average over epoch
            # total_train_err[epoch] = float(train_err) / total_epoch
    ##        total_train_loss[epoch] = float(train_loss) / (n + 1)
            # scheduler.step()
            train_loss = train_loss/(train_batches)#*batch_size)  
            print("epoch:",epoch,"loss_ita",train_loss)
            ita_list.append(train_loss)
            # if epoch%10==0:
            #     torch.save(model.state_dict(), os.path.join(args.output_dir,f"local_only__fold__{fold}__epoch__{epoch}__margin{args.margin}_similarity{args.similarity_measure}"))
            del loss0 ### Sajith 
            i2t_precision_tr=(i2t_corr_tr+t2i_corr_tr)/batch_epoch_tr
            # t2i_precision_tr=t2i_corr_tr/batch_epoch_tr
            print("training precision", i2t_precision_tr)##+t2i_precision_tr)/2)
            model.eval()
            with torch.set_grad_enabled(False):

                val_loss = 0.0
                val_b=0
                total_epoch = 0
                validation_true = []
                validation_estimated = []
                n = 0
                # valid_dl.dataset.dataset.test_or_val = True
                image_embeddings_val = []
                report_embeddings_val = []
    
                word_attention_val=[]
                word_embeddings_val=[]
                patch_embeddings_val=[]
                sent_list_val=[]
                patch_attention_list_val=[]
                cap_lens_list_val=[]
                patch_weights_val=[]
                word_weights_val=[]
                merged_att_val = []
                loc_val = []
                for num_batches_valid,(images, text,labells,masks,location) in enumerate(valid_dl):
                    
 #               for i, batch in enumerate(valid_dl):
     #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
      #              model.to(device)
            #_ = model(batch[0][0])
    #                if torch.cuda.is_available():
                    images= images.to(device)
                    mask = text['attention_mask'].to(device)
                    input_id = text['input_ids'].squeeze(1).to(device)
                    
                    # pred,img_emb_q,report_emb_q,patch_emb_q,word_feat_q,word_attn_q,sents,patch_atten_output,word_atten_output = model(images.float(),input_id, mask)
                    img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_atten_output_weight,word_atten_output_weight,cap_lens,patch_atten_output,word_atten_output,merged_att = model(images.float(),input_id, mask)
                    image_embeddings_val.append(img_emb_q)
                    report_embeddings_val.append(report_emb_q)
                    word_attention_val.append(word_atten_output)
                    word_embeddings_val.append(word_emb_q)
                    patch_embeddings_val.append(patch_emb_q)
                    sent_list_val.append(sents)
                    patch_attention_list_val.append(patch_atten_output)
                    cap_lens_list_val.append(cap_lens)
                    patch_weights_val.append(patch_atten_output_weight)
                    word_weights_val.append(word_atten_output_weight)
                    merged_att_val.append(merged_att)
                    loc_val.append(location)
                    # bz = img_emb_q.size(0)
                   
                    # prob = torch.sigmoid(pred)

                    # labs = torch.arange(bz).type_as(report_emb_q).long()
                    # labels = torch.eye(bz).type_as(report_emb_q)[labs]
                    # loss_fn = ContrastiveLoss_euclidean(margin=0.1)
                    # loss0 = loss_fn(img_emb_q,report_emb_q, labels)
                    # valid_loss = loss0
                    
                    
                    # for i in range(len(labells.tolist())):
                    #     validation_true.append(labells.tolist()[i])#[0])
                    #     validation_estimated.append(prob.tolist()[i])#[0])

                    if ((num_batches_valid + 1) % num_batch_accumulate == 0) or (num_batches_valid + 1 == len(valid_dl)): #If we have enough batches to take a step
                        
                        # val_batches = 0  
                        cap_len_list_val = [element for sublist in cap_lens_list_val for element in sublist]
                        val_batches+=1
                        image_embeddings_val = torch.cat(image_embeddings_val, dim=0)
                        report_embeddings_val = torch.cat(report_embeddings_val, dim=0)
                        word_attention_val=torch.cat(word_attention_val, dim=0)
                        word_embeddings_val=torch.cat(word_embeddings_val, dim=0)
                        patch_embeddings_val=torch.cat(patch_embeddings_val, dim=0)
                        # sent_list_val=torch.cat(sent_list_val, dim=0)
                        patch_attention_list_val=torch.cat(patch_attention_list_val, dim=0)
                        patch_weights_val=torch.cat(patch_weights_val, dim=0)
                        # sent_list_val=torch.cat(sent_list_val, dim=0)
                        word_weights_val=torch.cat(word_weights_val, dim=0)
                        merged_att_val=torch.cat(merged_att_val, dim=0)
                        loc_val=torch.cat(loc_val, dim=0)
                        bz = len(image_embeddings_val)
                   
                        # prob = torch.sigmoid(pred)

                        labs = torch.arange(bz).type_as(report_embeddings_val).long()
                        labels = torch.eye(bz).type_as(report_embeddings_val)[labs]

                        # euclidean_distance = torch.nn.functional.pairwise_distance(image_embeddings_val, report_embeddings_val, keepdim=True)
                        
                        # loss_fn = ContrastiveLoss_euclidean(margin=args.margin)
                        # if args.similarity_measure=="euclidian":
                        #     loss_fn = ContrastiveLoss_euclidean(margin=args.margin)
                        # else:
                        #     loss_fn ,i_t_scores,t_i_scores= ContrastiveLoss_cosine(margin=args.margin)
                        # valid_loss_global = loss_fn(image_embeddings_val,report_embeddings_val, labels)
##############################*************************** local_weights
                        # valid_loss_local_0 = local_contrastive_loss2(word_embeddings_val, patch_embeddings_val,patch_attention_list_val, cap_len_list_val,merged_att_val,margin=args.margin)
                        # valid_loss_local_1 = local_contrastive_loss2(patch_embeddings_val, word_embeddings_val,word_attention_val, cap_len_list_val,merged_att_val,margin=args.margin)

                        # # valid_loss_1 = loss_fn(report_embeddings_val,image_embeddings_val, labels)
                        # # valid_loss = (valid_loss_0 + valid_loss_1)/2
                        # # valid_loss = loss_fn(euclidean_distance, labels)
                        # # valid_loss = loss0
                        # valid_loss=valid_loss_local_0+valid_loss_local_1
                        # val_loss += valid_loss.detach().item()
                        ########################*******************************
                        
                        # i_t_scores=cosine_distance(patch_embeddings_val,patch_attention_list_val)[1]
                        i_t_scores = F.cosine_similarity(patch_embeddings_val.reshape(bz,-1).unsqueeze(1), patch_attention_list_val.reshape(bz,-1).unsqueeze(0), dim=-1)

                        i2t_acc1_val,i2t_corr_val_batch,i2t_batch_val = precision_at_k(i_t_scores)#, labels, top_k=(1,))
                        # t2i_acc1_val,t2i_corr_val_batch,t2i_batch_val = precision_at_k(t_i_scores, labels, top_k=(1,))
                        i2t_corr_val+=i2t_corr_val_batch
                        batch_epoch_val+=i2t_batch_val

                        image_embeddings_val=[]
                        report_embeddings_val=[]
                        word_attention_val=[]
                        word_embeddings_val=[]
                        patch_embeddings_val=[]
                        sent_list_val=[]
                        patch_attention_list_val=[]
                        cap_lens_list_val=[]
                        patch_weights_val=[]
                        word_weights_val=[]
                        merged_att_val = []
                        loc_val = []
    #                corr = (pred > 0.0).squeeze().long() != labels
                    # val_err += int(corr.sum())
    #                total_epoch += len(labels)
                    n = n + 1

                    ############******************
                # val_loss = val_loss / (val_batches)#* batch_size)
                # print("validation_ita",val_loss)
                # ita_list_val.append(val_loss)
                ################********
                i2t_precision_val=i2t_corr_val/batch_epoch_val
                # t2i_precision_val=t2i_corr_val/batch_epoch_val
                print("validation precision", i2t_precision_val)#+t2i_precision_val)/2
                
                # for images, labells in test_dl:
                        
                    
                #     images, labells = images.to(device), labells.to(device)
                #     pred = model(images)
                #     prob = torch.sigmoid(pred)#F.softmax(pred, dim=1)
                #     for i in range(len(labells.tolist())):
                #         test_true.append(labells.tolist()[i][0])
                #         test_estimated.append(prob.tolist()[i][0])

                # Calculate the AUC for the different models
    # #        print("sssssssssssssssssssssssss",validation_true)#len(validation_true))
    #         val_auc = roc_auc_score(validation_true, validation_estimated)#,multi_class='ovr')
    #         # test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
    #         train_auc = roc_auc_score(training_ture, training_estimated)#,multi_class='ovr')

                # total_val_err[epoch] = float(val_err) / total_epoch
    #            total_val_loss[epoch] = float(val_loss) / (n + 1)


        

            
            if epoch >=50 and epoch%10==0:
                print("ita_train_list",ita_list)
                print("ita_valid_list",ita_list_val)
            valid_loss=0
            del valid_loss
        model.eval()
        with torch.set_grad_enabled(False):

            # val_loss = 0.0
            # val_b=0
            # total_epoch = 0
            
            n = 0
            # test_dl.dataset.dataset.test_or_val = True
            
            for images, text,labells,masks,location in test_dl:
                
#               for i, batch in enumerate(valid_dl):
    #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
    #              model.to(device)
        #_ = model(batch[0][0])
#                if torch.cuda.is_available():
                images= images.to(device)
                mask = text['attention_mask'].to(device)
                input_id = text['input_ids'].squeeze(1).to(device)
        
                img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_atten_output_weight,word_atten_output_weight,cap_lens,patch_atten_output,word_atten_output,merged_att = model(images.float(),input_id, mask)#(**texts)#
               
                # pred = model(images)
                # def model_wrapper(*x):
                #     return model(*x)[0]
                # print(model) 
                # print("layer",model.layer4[0].conv2)               
                # layer_gc = LayerGradCam(model, model.layer4[0].conv2) #layer4[0].conv1
                # # layer_gc.grad_cam.forward_func = model_wrapper
                # # layer_gc.guided_backprop.forward_func = model_wrapper
                # attr = layer_gc.attribute(images)
                # print(model)
                # print("attr",attr.shape)
                
                ##$$$$upsampled_attr = LayerAttribution.interpolate(attr, (240,240,155))
                # print("attr",upsampled_attr.shape)
                ####$$$$numpy_heatmap=upsampled_attr .cpu().detach().numpy()
                # print(numpy_heatmap[0][0].shape)
                # for saving:
                # for i in numpy_heatmap:
                ###$$$$ np.save(os.path.join(args.output_dir,"model_heatmap.npy"),numpy_heatmap[0])#[i][0])
        
                
                # prob = torch.sigmoid(pred)#F.softmax(pred, dim=1)
                # loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                
                # for i in range(len(labells.tolist())):
                #     test_true.append(labells.tolist()[i])#[0])
                #     test_estimated.append(prob.tolist()[i])#[0])
                
            # test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
            # print("test_auc",test_auc)
        if fold==0:
            break

    #        train_fpr, train_tpr, _ = roc_curve(training_true, training_estimated)
    #        val_fpr, val_tpr, _ = roc_curve(validation_true, validation_estimated)

            

    #        logging.info(
    #            "Epoch {}: Train loss: {}, Train AUC: {} | Val loss: {}, Val AUC: {} ".format(
    #                epoch + 1,
    #                total_train_loss[epoch],
    #                total_train_auc[epoch],
    #                total_val_loss[epoch],
    #                total_val_auc[epoch]))

            # model_path = get_model_name(trial=fold, name=net.name, batch_size=batch_size, learning_rate=learning_rate,
                                        # dropout_rate=0.1, epoch=epoch + 1) #after mednet commented

        # if fold==3:
        # if 
                

        # torch.save(model.state_dict(), os.path.join(args.output_dir,f"_cl_attention__fold__{fold}"))

    # np.savetxt(os.path.join(save_folder, "{}_train_err.csv".format(model_path)), total_train_err)
#    np.savetxt(os.path.join(args.output_dir, "{}_train_loss.csv".format(model_path)), total_train_loss)
#    np.savetxt(os.path.join(args.output_dir, "{}_train_auc.csv".format(model_path)), total_train_auc)
    # np.savetxt(os.path.join(save_folder, "{}_val_err.csv".format(model_path)), total_val_err)
#    np.savetxt(os.path.join(args.output_dir, "{}_val_loss.csv".format(model_path)), total_val_loss)
#    np.savetxt(os.path.join(args.output_dir, "{}_val_auc.csv".format(model_path)), total_val_auc)



    logging.info('Finished training.')
#    logging.info(f'Time elapsed: {round(time.time() - training_start_time, 3)} seconds.')

    # return total_train_err, total_train_loss, total_train_auc, total_val_err, total_val_loss, total_val_auc
    return 0, total_train_auc, 0, total_val_auc

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False   
            
        



if __name__ == '__main__':
    print("without location")
    
    args = make_parser().parse_args()
    print("margin",args.margin)
    print("similarity",args.similarity_measure)
    print("weight_decay",args.weight_decay)
#    num_batch_accumulate = 16 / batch_size
#    random.seed(args.rseed)
#    np.random.seed(args.rseed)
#    torch.manual_seed(args.rseed)
#    cuda = torch.cuda.is_available() and args.gpus != '-1'
   # torch.backends.cudnn.deterministic = True
  #  torch.backends.cudnn.benchmark = True#False
  #  torch.backends.cudnn.enabled = False
#    if cuda:
#        torch.cuda.manual_seed(args.rseed)
#        torch.cuda.manual_seed_all(args.rseed)
#    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
#    torch.cuda.set_device("cuda:"+ args.gpus)
#    args.device = torch.device("cuda:"+ args.gpus) if cuda else torch.device('cpu')
#    logger.info(torch.cuda.get_device_name(args.device))
    image_folder=args.image_path###################################################################################3
    #####label_file=args.data_path#############3
    

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
#    df_sickkids = load_excel_data(os.path.join(radiomics_directory, 'Nomogram_study_LGG_data_Nov.27.xlsx'), sheet='SK')
    # SK_input,SK_labels = process_excel(label_file)
    # patients_included = SK_labels.keys()
    file = open('input_tokenized_text_all', 'rb') #important  #_if

# dump information to that file

    df_text_excel = pickle.load(file)

# close the file
    file.close()
    data=process_excel(df_text_excel)
    data.index=range(data.shape[0])
    # print(data.index)
    # print("columns",data.columns)
#    radiomics_patients_list = set(sickkids_labels.keys())
    # patients_with_FLAIR = []
#    for each_patient in os.listdir(image_):
#        try:
#        patients_with_FLAIR.append(int(each_patient))
#        except:
#            logging.info(f'Patient {each_patient} FLAIR not found.')
#    patients_with_FLAIR.sort(key=int)
#    patients_list = list(radiomics_patients_list.intersection(patients_with_FLAIR))
#    result_image_label = load_data_for_patient(image_folder, patients=patients_list, limit=limit)
   
    
    

    # for each_patient in range(len(patients_included)):
    #     result_image_label = load_data_for_patient(image_folder,list(patients_included)[each_patient])
        
    #     if result_image_label != None:
    #         data[list(patients_included)[each_patient]]= result_image_label    #change key
    #         patients_with_FLAIR.append(list(patients_included)[each_patient]) #CHANGE KEY

    # patients_to_use = patients_with_FLAIR


    #### model load

    # training_aucs = []
    # validation_aucs = []
    # test_aucs=[]
#    best_epochs = []
#    trial_times = []
    df_loc = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Nomogram_study_LGG_data_Nov.27.xlsx",engine='openpyxl')
    whole_data = BertDataset(data,image_folder, df_loc)#, patients_to_use)
    train_dataset,test_dataset = split_dataset_cv(whole_data,0.88)#0.8)#,0.6,0.2)#,0.6,0.2)
    file1 = open('test_dataset2', 'wb')

    # dump information to that file
    pickle.dump(test_dataset, file1)

    # close the file
    file1.close()
    file2 = open('train_dataset2', 'wb')

    # dump information to that file
    pickle.dump(train_dataset, file2)

    # close the file
    file1.close()
    test_dl=load_data(test_dataset,test_dataset,test_dataset, args.batch_size)[2]
    inplanes=[64, 128, 256, 512]#512*2]
    # net = generate_model(model_depth=18, inplanes=inplanes, n_classes=1039)#$$

    # net.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False) #$$
    # net.fc = net.fc = nn.Linear(512, 1)#3) #512  #$$

    # net.to(device)

    ##criterion = nn.BCEWithLogitsLoss()
    
    ##optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    batch_size=args.batch_size
    num_epochs=args.num_epochs
    learning_rate=learning_rate=args.lr
    criterion = nn.BCEWithLogitsLoss()#BCELoss()
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    if args.global_local=="global":  
        train_lossss, train_aucsss, val_lossss, val_aucsss=train_global_model(args,whole_data,test_dl,args.output_dir)

    elif args.global_local=="local":
        train_lossss, train_aucsss, val_lossss, val_aucsss=train_local_model(args,whole_data,test_dl,args.output_dir)
    
    else:
        train_lossss, train_aucsss, val_lossss, val_aucsss=train_global_local_model(args,whole_data,test_dl,args.output_dir)
                            











