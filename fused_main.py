# baseline MRI-based model
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
from datetime import datetime
from captum.attr import LayerGradCam,GuidedGradCam
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
# from fused_model_copy import image_text,image_text_attention
from fused_dataset import split_dataset_cv,process_excel,BertDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from fused_dataset import BertDataset,split_dataset_cv,process_excel,EvalDataset,Eval_new_sk, fewshot_support
from fused_model_copy import generate_model
from fused_model import ResNet_attention,downstream_image_classifier,SelfAttention, zeroshot_classifier, fewshot_classifier,image_text_attention, image_text, DepthAttention, MultiHeadSelfAttention, MultiHeadDepthAttention, LocalEmbedding_3d, BertClassifier
import pickle
import stat
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import StepLR
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from text_models import BertClassifier
from fused_main_global import count_trainable_parameters
plt.rcParams['figure.figsize'] = [25, 10]
# 651   629  673
logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('lgg_unet')
logging.getLogger('matplotlib.font_manager').disabled = True
pil_logger = logging.getLogger('PIL').setLevel(logging.INFO)

def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch SickKids Brain MRI')

    # Data
    parser.add_argument('--data_path', type=str, default="/hpf/largeprojects/fkhalvati/Sara/lgg/Nomogram_study_LGG_data_Nov.27.xlsx", help='Data path')
    parser.add_argument('--image_path', type=str, default="/hpf/largeprojects/fkhalvati/Projects/SickKids_Brain_Preprocessing/preprocessed_all_seq_kk_july_2022", help='image_path')
    parser.add_argument('--output_dir', type=str, default='/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text', help='Output directory')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('--resize', type=int, default=224, help='Resizing images')

    # Training
    parser.add_argument('--batch_size', type=int, default=4, help='batch size') #4  
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs') #0.0003
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate') #5e-2  #1e-3 #compare:5e-6 #6e-6 (best till now) 0.0002
    parser.add_argument('--scheduler', default=False, action='store_true', help='[USE] scheduler') ##true?
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')

    
    parser.add_argument('--dropout', type=float, default=0, help='dropout') #0.1 #0.15

    # Misc
    parser.add_argument('--gpus', type=str, default='0', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test [same as train if not set]')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--weight_epoch', type=int)
    parser.add_argument('--stage', type=str , default = "training")
    return parser

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value) 
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

random_seed(14, True)

splits=KFold(n_splits=5,shuffle=True,random_state=42)


def model_eval(model,valid_dl,device="cuda"):

    model.eval()
    model=model.cuda()
    with torch.set_grad_enabled(False):

        val_loss = 0.0
        val_b=0
        
        validation_true = []
        validation_estimated = []
        n = 0
        classifier_criterion = nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()#nn.BCEWithLogitsLoss()
        
        for images,texts, labells in valid_dl:
#              
            images = images.to(device)    
            labells = labells.to(device)
            # mask = texts['attention_mask'].to(device)
            # input_id = texts['input_ids'].squeeze(1).to(device)
            pred = model(input_id, mask)               
            
            prob = torch.sigmoid(pred)#F.softmax(pred, dim=1)#torch.sigmoid(pred)#F.softmax(pred, dim=1)
            # pred_class=prob.max(1, keepdim=True)[0]#prob.argmax(dim=1)
            loss = classifier_criterion(pred.squeeze(-1),labells.float())#(pred.squeeze(-1), labells.float())   
            for i in range(len(labells.tolist())):
                validation_true.append(labells.tolist()[i])#[0])
                validation_estimated.append(prob.tolist()[i])#[0]) #pred_class


            val_loss += loss.item()
            
    val_auc = roc_auc_score(validation_true, validation_estimated)
    # print("test AUC:", val_auc)
    return val_auc

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
        return total_loss



def train_model(args, data_ds,test_dl, output_model_path, tuning=False):
    
    global_step = 0
    softmax_temperature: float = 0.07
    local_temperature=0.1
    # 1 1 0.125
    # lambda_1: float =1#0.5# 0.5#1#1#0.7
    # lambda_2: float =1#0.4
    #0.34 #0.5#0.7
    # lambda_3: float = 0.125#0.6#0.66#0.7#0.5#0.5
    num_heads = 8
    emb_dim = 128
    num_prototypes= 2
    
 #   optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # clr = cyclizcal_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
    batch_size=16
    num_batch_accumulate = 4/batch_size #128 / batch_size
  #  val_batch_accumulate =  128/batch_size
  #  gradcam_dir = os.path.join(args.output_dir,"gradcam_dir")
 #   model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True,label=1) #???
 #   print(medcam.get_layers(model))
 #   cam = GradCAM(nn_module=model, target_layers=model.fc)#"class_layers.relu")
#    scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])
    classifier_criterion = nn.BCEWithLogitsLoss()#BCELoss()#.
    
    test_auc=[]

    weight_path="/hpf/largeprojects/fkhalvati/Sara/pretrain/resnet_18.pth"
    best_auc=0
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_ds)))):
    #    print(train_idx,val_idx)
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_dl = DataLoader(data_ds, batch_size=batch_size, sampler=train_sampler)
        valid_dl = DataLoader(data_ds, batch_size=batch_size, sampler=test_sampler)
        model=image_text_attention(emb_dim=128,num_heads=num_heads)
        # opt={}
        ##################3pretrain
        # model = generate_model(model_depth=18, inplanes=inplanes, n_classes=1039)
        # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # net_dict = model.state_dict()
        
        # pretrain = torch.load(weight_path)
        # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        # net_dict.update(pretrain_dict)
        # model.load_state_dict(net_dict)
        ##############3pretrain
        # model.load_state_dict(weight_path['state_dict'])  ####mednet
        # optimizer.load_state_dict(weight_path['optimizer']) ### mednet
        # for p in parameters:
        # for pname, p in model.named_parameters():
        #     p.requires_grad = False
        # for p in model.layer4.parameters():
        #     p.requires_grad = True
        # for p in model.layer3.parameters():
        #     p.requires_grad = True
        # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # patch_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # word_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # model.fc  = nn.Linear(512, 1)#nn.Sequential(nn.Linear(512, 256),nn.Linear(256, 1))#3) #512

        model.to(device)
        # patch_local_atten_layer.to(device)
        # word_local_atten_layer.to(device)
        loss_fn = DynamicWeightedLoss(num_losses=3)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=400,T_mult=1, eta_min=1e-8, last_epoch=-1)
            
        #(optimizer, step_size=10, gamma=0.1)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        optimizer.zero_grad()
        ita_list=[]
        local_list=[]
        cl_list=[]
        for epoch in range(args.num_epochs):
            model.train()
            train_loss = 0
            counter = 0
            num_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]
     #       print(len(train_dl))
            for num_batches,(images,text, labells,masks) in enumerate(train_dl): 
                # print("NEW BATCH")
   #             print("NNNN",ba)S
    #            print()
    #           print(labells)
    #          break
   #             num_batches+=1#1
                
                images ,labells= images.to(device),labells.to(device)
                # text=text.cuda()
                mask = text['attention_mask'].to(device)
                input_id = text['input_ids'].squeeze(1).to(device)
  #              model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True,label=1)
            # labells = labells.cuda()
                #noise???

    #            loss_classifier = classifier_criterion(y_pred, labels)

    #            epoch_loss += total_loss.item()

                
                model.to(device)
         #       optimizer.zero_grad()
                # pred=model(images)
                images=images.float()
                pred,img_emb_q,report_emb_q,patch_emb_q,word_emb_q,word_attn_q,sents,patch_atten_output,word_atten_output= model(images,input_id, mask)
                prob = torch.sigmoid(pred)
                
                #instance-wise loss
                bz = img_emb_q.size(0)
                labels = torch.arange(bz).type_as(report_emb_q).long()
                scores = img_emb_q.mm(report_emb_q.t())
                scores /= softmax_temperature
                scores1 = scores.transpose(0, 1)
                loss0 = F.cross_entropy(scores, labels)
                loss1 = F.cross_entropy(scores1, labels)
                loss_ita = loss0 + loss1

                #################### token_pixel alignment
                mask_pad = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(images).bool()
                # word_atten_output, _ = word_local_atten_layer(word_emb_q, patch_emb_q, patch_emb_q,need_weights=True)
                # word_atten_output = F.normalize(word_atten_output, dim=-1)

                word_sim = torch.bmm(word_emb_q, word_atten_output.permute(0, 2, 1)) / local_temperature
                ##################### local_loss
                with torch.no_grad():
                    # print("word_atttn", word_attn_q)
                    atten_weights = word_attn_q.detach()
                    word_atten_weights = []
                    for i in range(bz):
                        atten_weight = atten_weights[i]
                        nonzero = atten_weight.nonzero().squeeze()
                        low = torch.quantile(atten_weight[nonzero], 0.1)
                        high = torch.quantile(atten_weight[nonzero], 0.9)
                        atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                        word_atten_weights.append(atten_weight.clone())
                    word_atten_weights = torch.stack(word_atten_weights)
            # maybe clip the tensor of 10 percentile and 90 percentile

                word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)
                ######## loss_local
                word_sim = torch.bmm(word_emb_q, word_atten_output.permute(
                    0, 2, 1)) / local_temperature
                word_num = word_sim.size(1)
                word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
                targets = torch.arange(word_num).type_as(
                    word_emb_q).long().repeat(bz)
                loss_word_1 = torch.sum(F.cross_entropy(
                    # word_sim_1, targets, reduction="none") ) / bz
                    word_sim_1, targets, reduction="none") * word_atten_weights.view(-1)) / bz
                # loss_word_1 = torch.sum(F.cross_entropy(
                #     word_sim_1, targets, reduction="none") ) / bz

                word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
                loss_word_2 = torch.sum(F.cross_entropy(
                    # word_sim_2, targets, reduction="none") ) / bz
                    word_sim_2, targets, reduction="none") * word_atten_weights.view(-1)) / bz
                # loss_word_2 = torch.sum(F.cross_entropy(
                #     word_sim_2, targets, reduction="none") ) / bz

                loss_word = (loss_word_1 + loss_word_2) / 2.




                # patch_atten_output, _ = patch_local_atten_layer(patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask_pad,need_weights=True)
                # patch_atten_output = F.normalize(patch_atten_output, dim=-1)
                patch_num = patch_atten_output.size(1)
                patch_atten_weights = torch.ones(bz, patch_num).type_as(images) / patch_num

                patch_sim = torch.bmm(patch_emb_q, patch_atten_output.permute(
                0, 2, 1)) / local_temperature
                patch_num = patch_sim.size(1)
                patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
                targets = torch.arange(patch_num).type_as(
                    patch_emb_q).long().repeat(bz)
                # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
                loss_patch_1 = torch.sum(F.cross_entropy(
                    patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

                patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
                loss_patch_2 = torch.sum(F.cross_entropy(
                    patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

                loss_patch = (loss_patch_1 + loss_patch_2) / 2.

                loss_local = loss_patch + loss_word

                # lossses only global
                # loss_classification=classifier_criterion(pred,labells.unsqueeze(1).float())  #.float()
                # losses = torch.stack([loss_classification, loss_local, loss_ita], dim=0)
                # loss = loss_fn(losses)
                #losses only global
                loss=loss_ita



                # loss= lambda_1*loss_classification + lambda_2 * loss_ita + lambda_3 *loss_local 

                loss.backward()#(retain_graph=True) # retain_graph=True done for tie net model
                # -- clip the gradient at a specified value in the range of [-clip, clip].
                # -- This is mainly used to prevent exploding or vanishing gradients in the network training
    #            nn.utils.clip_grad_value_(model.parameters(), 0.1)
       #         if num_batches==num_batch_accumulate: #If we have enough batches to take a step
                if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                    optimizer.step() #after if 
                    optimizer.zero_grad() # after if
                    if num_batches==30:
                        # print("num_batch",num_batches)
                        print("loss_ita",loss_ita)
                        # print("loss_local",loss_local)
                        # print("loss_classification",loss_classification)
                        ita_list.append(loss_ita)
                        local_list.append(loss_local)
                        # cl_list.append(loss_classification)
                
                for i in range(len(labells.tolist())):
                    # print("ll",labells)
                    training_ture.append(labells.tolist()[i])#[0])
                    training_estimated.append(prob.tolist()[i])#[0]) #prob
               #     model_grad = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)

    #            corr = (pred>0.0).squeeze().long() != labells
                train_loss+=loss.item()*batch_size
                #?
                train_batches+=1
                counter += 1

            # Calculate average over epoch
            # total_train_err[epoch] = float(train_err) / total_epoch
    ##        total_train_loss[epoch] = float(train_loss) / (n + 1)
            # scheduler.step()
            train_loss = train_loss/(train_batches*batch_size)  
            
            model.eval()
            with torch.set_grad_enabled(False):

                val_loss = 0.0
                val_b=0
                total_epoch = 0
                validation_true = []
                validation_estimated = []
                n = 0
                # valid_dl.dataset.dataset.test_or_val = True
                
                for images, text,labells in valid_dl:
                    
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
                    loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                    
                    for i in range(len(labells.tolist())):
                        validation_true.append(labells.tolist()[i])#[0])
                        validation_estimated.append(prob.tolist()[i])#[0])

    #                if val_batches==num_batch_accumulate: #If we have enough batches to take a step
                        
    #                    val_batches = 0  


                    val_loss += loss.item()
    #                corr = (pred > 0.0).squeeze().long() != labels
                    # val_err += int(corr.sum())
    #                total_epoch += len(labels)
                    n = n + 1
    #           val_loss = val_loss / (batches * batch_size)
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
        print("ita_list",ita_list)
        print("local_list",local_list)
        print("cl_list",cl_list)
        model.eval()
        with torch.set_grad_enabled(False):

            # val_loss = 0.0
            # val_b=0
            # total_epoch = 0
            test_true = []
            test_estimated = []
            n = 0
            # test_dl.dataset.dataset.test_or_val = True
            
            for images, text,labells in test_dl:
                
#               for i, batch in enumerate(valid_dl):
    #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
    #              model.to(device)
        #_ = model(batch[0][0])
#                if torch.cuda.is_available():

                images, labells = images.to(device), labells.to(device)
                mask = text['attention_mask'].to(device)
                input_id = text['input_ids'].squeeze(1).to(device)
        
                pred,image_emb_q,report_emb_q,patch_emb_q,word_feat_q,word_attn_q,sents,patch_atten_output,word_atten_output = model(images.float(),input_id, mask)#(**texts)#

                prob = torch.sigmoid(pred)#F.softmax(pred, dim=1)
                # loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                
                for i in range(len(labells.tolist())):
                    test_true.append(labells.tolist()[i])#[0])
                    test_estimated.append(prob.tolist()[i])#[0])
                if epoch%100==0:
                    torch.save(model.state_dict(), os.path.join(args.output_dir,f"only_global__fold__{fold}"))
            test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
            print("test_auc",test_auc)
        # if fold==0:
        #     break

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
            





def train_image_model_cv(args, data_ds, output_model_path, tuning=False):
    
    
    print("training started")
    
 #   optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
    batch_size=args.batch_size
    num_batch_accumulate = 16/batch_size #128 / batch_size

    classifier_criterion = nn.BCEWithLogitsLoss()#BCELoss()#.
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
#    optimizer.zero_grad()

    weight_path="/hpf/largeprojects/fkhalvati/Sara/pretrain/resnet_18_23dataset.pth"
    best_auc=0
    outer_kfold = KFold(n_splits=5, shuffle=True,random_state=42)
    for test_fold, (outer_train_indices,outer_test_indices) in enumerate(outer_kfold.split(data_ds)):
        
        print(f"test_fold:{test_fold}")
        outer_train_set = torch.utils.data.Subset(data_ds, outer_train_indices)
        outer_test_set = torch.utils.data.Subset(data_ds, outer_test_indices)
    
        
        test_dl = DataLoader(outer_test_set, batch_size=batch_size)
        inner_kfold = KFold(n_splits=5, shuffle=True,random_state=42)

        # file = open(f'test_dataset_MRI{test_fold}', 'wb')

        # # # dump information to that file
        # pickle.dump(outer_test_set, file)

        # # # close the file
        # file.close()
        
        
        for valid_fold, (inner_train_indices, inner_val_indices) in enumerate(splits.split(outer_train_set)):

            print(f"valid_fold:{valid_fold}")

            inner_train_set = torch.utils.data.Subset(outer_train_set, inner_train_indices)
            inner_val_set = torch.utils.data.Subset(outer_train_set, inner_val_indices)

            

            file = open(f'train_dataset_MRI{test_fold,valid_fold}', 'wb')

            # # dump information to that file
            pickle.dump(inner_train_set, file)

            # # close the file
            file.close()


            file = open(f'valid_dataset_MRI{test_fold,valid_fold}', 'wb')

            # # dump information to that file
            pickle.dump(inner_val_set, file)
            
            # # close the file
            file.close()
            print("donneee")
            train_dl =DataLoader(inner_train_set, batch_size=batch_size, shuffle=True)        
            valid_dl = DataLoader(inner_val_set, batch_size=batch_size, shuffle=True)
            inplanes=[64, 128, 256, 512]

            model=downstream_image_classifier()
            # model=generate_model(model_depth=18, n_classes=1039,inplanes=inplanes)
            # model.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1 , 2, 2), padding=(3, 3, 3), bias=False)
            # del model.cnn.local_embed
            
            ##################3pretrain
            # model = generate_model(model_depth=18, inplanes=inplanes, n_classes=1039)
            # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
            net_dict = model.state_dict()
            
            pretrain = torch.load(weight_path)
            # print("pretrain",pretrain)#["layer4.1.conv2.weight"])
            pretrain['state_dict'] = {k.replace('module.', 'cnn.'): v for k, v in pretrain['state_dict'].items()}
            # print("transfer learning",[k for k, v in pretrain['state_dict'].items()])
            # print("nettt dictt", net_dict.keys())
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items()}# if k in net_dict.keys()}#***
            # print("TLLLLLLLL",[k for k, v in pretrain['state_dict'].items()])
            # print("pretrain",pretrain_dict)

            # net_dict.update(pretrain_dict)#***

            # print("net_dict,", net_dict["layer4.1.conv2.weight"])



            model.load_state_dict(pretrain_dict)
            # model.load_state_dict(net_dict)#***
            ##############3pretrain


            #rename

            model.cnn.fc = nn.Linear(512, 256)
            model.cnn.fc1 = nn.Linear( 256,1)

            # model.load_state_dict(weight_path['state_dict'])  ####mednet
            # model.load_state_dict(pretrain) 
            # optimizer.load_state_dict(weight_path['optimizer']) ### mednet
            # for p in parameters:

            model.cnn.self_attention = SelfAttention(512)#(512)
            # model.cnn.self_attention2 = SelfAttention()
            # model.cnn.depth_attention = DepthAttention(512)
            
            # model.cnn.self_attention2 = SelfAttention(256)
            # model.cnn.depth_attention2 = DepthAttention(256)

            for pname, p in model.cnn.named_parameters():
                p.requires_grad = False
            for  p in model.cnn.fc.parameters():
                p.requires_grad = True
            # for p in model.layer4.parameters():
            #     p.requires_grad = True
            for p in model.cnn.layer3.parameters():
                p.requires_grad = True
            # for p in model.cnn.layer2.parameters():
            #     p.requires_grad = True
            for p in model.cnn.layer4.parameters():
                p.requires_grad = True
            for p in model.cnn.fc1.parameters():
                p.requires_grad = True
            for p in model.cnn.self_attention.parameters():
                p.requires_grad = True
            # for p in model.cnn.self_attention2.parameters():
            #     p.requires_grad = True
            # for p in model.cnn.depth_attention.parameters():
            #     p.requires_grad = True

            # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
            # patch_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
            # word_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
            # model.fc  = nn.Linear(512, 1)#nn.Sequential(nn.Linear(512, 256),nn.Linear(256, 1))#3) #512

            model.to(device)
            
            # model.eval()



            # loss_fn = DynamicWeightedLoss(num_losses=3)

            # parameters = []
            # for name, param in model.named_parameters():
            #     if 'layer4' in name or 'fc' in name:
            #         parameters.append({'params': param, 'weight_decay': 0.001})
            #     else:
            #         parameters.append({'params': param, 'weight_decay': 0.0})
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)#,weight_decay=0.001)
            # print("pretrain",pretrain.keys())
            # optimizer.load_state_dict(pretrain['optimizer'])
            # optimizer = optim.Adam(parameters, lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            # scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=400,T_mult=1, eta_min=1e-8, last_epoch=-1)
                
            #(optimizer, step_size=10, gamma=0.1)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            optimizer.zero_grad()
            cl_list=[]
            for epoch in range(args.num_epochs):
                model.train()
                # # model.layer1.eval()
                # # model.layer2.eval()
                # # model.layer3.eval()
                # # model.eval()
                # model.layer4.train()
                # model.fc.train()
                # model.layer3.train()
                train_loss = 0
                counter = 0
                num_batches=0
                val_batch=0
                train_batches=0
                epoch_loss = 0
                training_ture=[]
                training_estimated=[]
    
                for num_batches,(images, text,labells,_, _) in enumerate(train_dl): #,masks,location
            
                    images ,labells= images.to(device),labells.to(device)
                    
    


        #            epoch_loss += total_loss.item()

                    
                    model.to(device)
            #       optimizer.zero_grad()
                    images=images.float()
                    pred , _ = model(images)
                    # pred= model(images)
                    prob = torch.sigmoid(pred)

                    # print("pred",pred.shape,labells.shape)
                    loss=classifier_criterion(pred,labells.unsqueeze(1).float())  
                    
                    
                

                    loss.backward()#(retain_graph=True) # retain_graph=True done for tie net model
                    # -- clip the gradient at a specified value in the range of [-clip, clip].
                    # -- This is mainly used to prevent exploding or vanishing gradients in the network training
        #            nn.utils.clip_grad_value_(model.parameters(), 0.1)
        #         if num_batches==num_batch_accumulate: #If we have enough batches to take a step
                    if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                        optimizer.step() #after if 
                        optimizer.zero_grad() # after if
                    
                    for i in range(len(labells.tolist())):
                        training_ture.append(labells.tolist()[i])#[0])
                        training_estimated.append(prob.tolist()[i])#[0]) #prob
                
                    train_loss+=loss.item()*batch_size
                    #?
                    train_batches+=1
                    counter += 1

                train_loss = train_loss/(train_batches)
                if epoch==19:
                    torch.save(model.state_dict(), os.path.join(args.output_dir,f"ROI_base_MRI_classification__resnet18_23_valid_fold__{test_fold}{valid_fold}.pth"))
                scheduler.step()
                model.eval()
                with torch.set_grad_enabled(False):

                    val_loss = 0.0
                    val_b=0
                    total_epoch = 0
                    validation_true = []
                    validation_estimated = []
                    
                    for val_batches,(images,text,labells,_,_) in enumerate(valid_dl):  #,masks,location
                        
                        images, labells = images.to(device), labells.to(device)
                        
            
                        pred ,_= model(images.float())
                        # pred= model(images.float())
                        
                
                        
                        prob = torch.sigmoid(pred)
                        loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                        
                        for i in range(len(labells.tolist())):
                            validation_true.append(labells.tolist()[i])#[0])
                            validation_estimated.append(prob.tolist()[i])#[0])

                    
            
                val_auc = roc_auc_score(validation_true, validation_estimated)#,multi_class='ovr')
                
                train_auc = roc_auc_score(training_ture, training_estimated)#,multi_class='ovr')

                


            

                total_train_auc[epoch] = train_auc
                total_val_auc[epoch] = val_auc
                
                print("epoch",epoch,":","train_AUC:",train_auc,"val_AUC",val_auc)
                if epoch==args.num_epochs:
                    if val_auc>max_auc:
                        max_auc=val_auc
                        best_model=model

            model.eval()
            with torch.set_grad_enabled(False):

                
                test_true = []
                test_estimated = []
                
                for images,text,labells,_, _ in test_dl:  
                    

                    images, labells = images.to(device), labells.to(device)
                    
            
                    pred ,_ = model(images.float())
                    # pred= model(images.float())
                
                    prob = torch.sigmoid(pred)
                    for i in range(len(labells.tolist())):
                        test_true.append(labells.tolist()[i])#[0])
                        test_estimated.append(prob.tolist()[i])#[0])
                    # if epoch==args.num_epochs-1:
                    #     print("test AUC:", model_eval(model,test_dl))
            
                test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
                print("test_auc",test_auc)
                break
        # if test_fold==0:
        #     break


    logging.info('Finished training.')
    return 0, total_train_auc, 0, total_val_auc






def train_image_model(args, data_ds,test_dl, output_model_path, tuning=False):
    
    

    
 #   optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
    batch_size=args.batch_size
    num_batch_accumulate = 16/batch_size #128 / batch_size

    classifier_criterion = nn.BCEWithLogitsLoss()#BCELoss()#.
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
#    optimizer.zero_grad()

    weight_path="/hpf/largeprojects/fkhalvati/Sara/pretrain/resnet_18.pth"
    best_auc=0
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_ds)))):
    #    print(train_idx,val_idx)
    #    print(f"trial{t} started")
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_dl = DataLoader(data_ds, batch_size=batch_size, sampler=train_sampler)
        valid_dl = DataLoader(data_ds, batch_size=batch_size, sampler=test_sampler)
        inplanes=[64, 128, 256, 512]
        # inplanes=[32,64, 128, 256]

        model=generate_model(model_depth=18, n_classes=1039,inplanes=inplanes)
        model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        model.fc = nn.Linear(512, 1)
        # opt={}
        ##################3pretrain
        # model = generate_model(model_depth=18, inplanes=inplanes, n_classes=1039)
        # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        net_dict = model.state_dict()
        
        pretrain = torch.load(weight_path)
        # print("pretrain",pretrain)#["layer4.1.conv2.weight"])
        pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}#***
        
        # print("pretrain",pretrain_dict)
        net_dict.update(pretrain_dict)#***
        # print("net_dict,", net_dict["layer4.1.conv2.weight"])
        model.load_state_dict(net_dict)#***
        ##############3pretrain


        #rename



        # model.load_state_dict(weight_path['state_dict'])  ####mednet
        # model.load_state_dict(pretrain) 
        # optimizer.load_state_dict(weight_path['optimizer']) ### mednet
        # for p in parameters:
        for pname, p in model.named_parameters():
            p.requires_grad = False
        for  p in model.fc.parameters():
            p.requires_grad = True
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.layer3.parameters():
            p.requires_grad = True
        # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # patch_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # word_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # model.fc  = nn.Linear(512, 1)#nn.Sequential(nn.Linear(512, 256),nn.Linear(256, 1))#3) #512

        model.to(device)
        
        # model.eval()
        # loss_fn = DynamicWeightedLoss(num_losses=3)

        # parameters = []
        # for name, param in model.named_parameters():
        #     if 'layer4' in name or 'fc' in name:
        #         parameters.append({'params': param, 'weight_decay': 0.001})
        #     else:
        #         parameters.append({'params': param, 'weight_decay': 0.0})
        optimizer = optim.Adam(model.parameters(), lr=args.lr)#,weight_decay=args.weight_decay)
        # optimizer = optim.Adam(parameters, lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=400,T_mult=1, eta_min=1e-8, last_epoch=-1)
            
        #(optimizer, step_size=10, gamma=0.1)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        optimizer.zero_grad()
        cl_list=[]
        for epoch in range(args.num_epochs):
            # model.train()
            # model.eval()
            model.layer4.train()
            model.fc.train()
            model.layer3.train()
            train_loss = 0
            counter = 0
            num_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]
   
            for num_batches,(images, text,labells) in enumerate(train_dl): 
        
                images ,labells= images.to(device),labells.to(device)
                
  


    #            epoch_loss += total_loss.item()

                
                model.to(device)
         #       optimizer.zero_grad()
                images=images.float()
                pred= model(images)
                prob = torch.sigmoid(pred)

                # print("pred",pred.shape,labells.shape)
                loss=classifier_criterion(pred,labells.unsqueeze(1).float())  
                
                
               

                loss.backward()#(retain_graph=True) # retain_graph=True done for tie net model
                # -- clip the gradient at a specified value in the range of [-clip, clip].
                # -- This is mainly used to prevent exploding or vanishing gradients in the network training
    #            nn.utils.clip_grad_value_(model.parameters(), 0.1)
       #         if num_batches==num_batch_accumulate: #If we have enough batches to take a step
                if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                    optimizer.step() #after if 
                    optimizer.zero_grad() # after if
                
                for i in range(len(labells.tolist())):
                    training_ture.append(labells.tolist()[i])#[0])
                    training_estimated.append(prob.tolist()[i])#[0]) #prob
            

    #            corr = (pred>0.0).squeeze().long() != labells
                train_loss+=loss.item()*batch_size
                #?
                train_batches+=1
                counter += 1

            # Calculate average over epoch
            # total_train_err[epoch] = float(train_err) / total_epoch
    ##        total_train_loss[epoch] = float(train_loss) / (n + 1)
            # scheduler.step()
            train_loss = train_loss/(train_batches)#*batch_size)  
            
            model.eval()
            with torch.set_grad_enabled(False):

                val_loss = 0.0
                val_b=0
                total_epoch = 0
                validation_true = []
                validation_estimated = []
                n = 0
                # valid_dl.dataset.dataset.test_or_val = True
                
                for val_batches,(images,text,labells) in enumerate(valid_dl):
                    
                    images, labells = images.to(device), labells.to(device)
                    
           
                    pred = model(images.float())
 
                    
            
                    
                    prob = torch.sigmoid(pred)
                    loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                    
                    for i in range(len(labells.tolist())):
                        validation_true.append(labells.tolist()[i])#[0])
                        validation_estimated.append(prob.tolist()[i])#[0])

    #                 if ((num_batches_valid + 1) % num_batch_accumulate == 0) or (num_batches_valid + 1 == len(valid_dl)): #If we have enough batches to take a step
                        
    # #                    val_batches = 0  


    #                 val_loss += loss.item()
    # #                corr = (pred > 0.0).squeeze().long() != labels
    #                 # val_err += int(corr.sum())
    # #                total_epoch += len(labels)
    #                 n = n + 1
    #           val_loss = val_loss / (val_batches)# * batch_size)
                
        
            val_auc = roc_auc_score(validation_true, validation_estimated)#,multi_class='ovr')
            
            train_auc = roc_auc_score(training_ture, training_estimated)#,multi_class='ovr')

               


        

            total_train_auc[epoch] = train_auc
            total_val_auc[epoch] = val_auc
            ##### self-added
            print("epoch",epoch,":","train_AUC:",train_auc,"val_AUC",val_auc)
        model.eval()
        with torch.set_grad_enabled(False):

            
            test_true = []
            test_estimated = []
            n = 0
            # test_dl.dataset.dataset.test_or_val = True
            
            for images,text,labells in test_dl:
                

                images, labells = images.to(device), labells.to(device)
                
        
                pred = model(images.float())
            
                prob = torch.sigmoid(pred)
                for i in range(len(labells.tolist())):
                    test_true.append(labells.tolist()[i])#[0])
                    test_estimated.append(prob.tolist()[i])#[0])
                if epoch%100==0:
                    torch.save(model.state_dict(), os.path.join(args.output_dir,f"whole_MRI_classification__fold__{fold}"))
            test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
            print("test_auc",test_auc)
        # if fold==0:
        #     break


    logging.info('Finished training.')
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
            
        
def train_downstream_image_model_cv(args, data_ds, output_model_path, tuning=False):
    
    

    print("here")
    
 #   optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
    batch_size=args.batch_size
    num_batch_accumulate = 16/batch_size #128 / batch_size

    classifier_criterion = nn.BCEWithLogitsLoss()#BCELoss()#.
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
#    optimizer.zero_grad()
    # weight_path = "/hpf/largeprojects/fkhalvati/Sara/pretrain/resnet_18_23dataset.pth"
    # weight_path = f"/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text/revised_attention_local_global__fold__0__epoch__{args.weight_epoch}__margin0.25"
    # weight_path = f"/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text/_na_location_attention_local_global__fold__0__epoch__{args.weight_epoch}__margin0.25" #_updated
    weight_path = f"/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text/_without_location_if_updated_location_attention_local_global__fold__0__epoch__340__margin0.25"
    # weight_path =os.path.join(args.output_dir,f"downstream_whole_MRI_classification__valid_fold__lr0.0003_{4}{0}.pth")
    # weight_path=f"/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text/_location_attention_local_global__fold__0__epoch__{args.weight_epoch}__margin0.25"
    best_auc=0
    outer_kfold = KFold(n_splits=5, shuffle=True,random_state=42)
    for test_fold, (outer_train_indices,outer_test_indices) in enumerate(outer_kfold.split(data_ds)):
        if test_fold<2:# or test_fold==2 or test_fold==3:
            continue
        print(f"test_fold:{test_fold}")
        print("newwwwwwwwwwwwwwww")
        outer_train_set = torch.utils.data.Subset(data_ds, outer_train_indices)
        # outer_train_set = data_ds
        outer_test_set = torch.utils.data.Subset(data_ds, outer_test_indices)
    
        
        test_dl = DataLoader(outer_test_set, batch_size=batch_size)
        inner_kfold = KFold(n_splits=5, shuffle=True,random_state=42)

        file = open('test_dataset_MRI', 'wb')

        # # dump information to that file
        pickle.dump(outer_test_set, file)

        # # close the file
        file.close()
        
        for valid_fold, (inner_train_indices, inner_val_indices) in enumerate(splits.split(outer_train_set)):

            print(f"valid_fold:{valid_fold}")

            inner_train_set = torch.utils.data.Subset(outer_train_set, inner_train_indices)
            inner_val_set = torch.utils.data.Subset(outer_train_set, inner_val_indices)

            

            # file = open('train_dataset_MRI', 'wb')

            # # # dump information to that file
            # pickle.dump(inner_train_set, file)

            # # # close the file
            # file.close()


            # file = open('valid_dataset_MRI', 'wb')

            # # # dump information to that file
            # pickle.dump(inner_val_set, file)

            # # # close the file
            # file.close()
            
            train_dl =DataLoader(inner_train_set, batch_size=batch_size, shuffle=True)        
            valid_dl = DataLoader(inner_val_set, batch_size=batch_size, shuffle=True)
            # inplanes=[64, 128, 256, 512]
            # inplanes=[32,64, 128, 256]

            model=downstream_image_classifier()
            model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
            # model.cnn.fc = nn.Linear(1039, 512)
            model.cnn.self_attention = SelfAttention(512)
            ##################3pretrain
            
            
            # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
            net_dict = model.cnn.state_dict()
            
            pretrain = torch.load(weight_path)
            # print("pretrain",pretrain)#["layer4.1.conv2.weight"])
            # pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
            # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items()}# if k in net_dict.keys()}#***
            
            # # print("pretrain",pretrain_dict)
            # net_dict.update(pretrain_dict)#***
            # print("net_dict,", net_dict["layer4.1.conv2.weight"])
            # del model.cnn.global_embed
            # del model.cnn.local_embed
            
            pretrain_cnn={k: v for k, v in pretrain.items() if k.startswith("cnn") and  "local_embed" not in k and "global_embed" not in k and "depth_attention" not in k}# and "self_attention" not in k}# and "local_embed.conv1d_1" not in k and "local_embed.conv1d_2" not in k}# and "local_embed" not in k and "global_embed" not in k }#and "fc" not in k}
            pretrain_cnn = {k.replace('cnn.', ''): v for k, v in pretrain_cnn.items()}
            # print("modellllllllllll",model.cnn.state_dict().keys())
            model.cnn.load_state_dict(pretrain_cnn)#***
            # net_dict.update(pretrain_cnn)
            ##############3pretrain

            # resnet = model.cnn
            # tsne = TSNE(n_components=2, random_state=42)
            
            model.cnn.fc=nn.Linear(512, 256)
            model.cnn.fc1=nn.Linear(256,1)
            # model.cnn.self_attention = SelfAttention(512)

            # model.cnn.fc=nn.Linear(320, 128)
            # model.cnn.fc1=nn.Linear(128,1)

            # model.cnn.self_attention = nn.MultiheadAttention(
            # embed_dim=512,num_heads=1, batch_first=True,dropout=0.5)
            # model.cnn.self_attention = SelfAttention(512)
            
            #rename

            # for name, module in model.named_children():
            #     print("module",name, module)     

            # model.load_state_dict(weight_path['state_dict'])  ####mednet
            # model.load_state_dict(pretrain) 
            # optimizer.load_state_dict(weight_path['optimizer']) ### mednet
            # for p in parameters:
            # del model.cnn.global_embed
            # del model.cnn.local_embed
            for pname, p in model.cnn.named_parameters():
                p.requires_grad = False
            for pname, p in model.cnn.fc.named_parameters():
                p.requires_grad = True
            for  p in model.cnn.layer4.parameters():
                p.requires_grad = True
            for  p in model.cnn.layer3.parameters():
                p.requires_grad = True
            for  p in model.cnn.fc1.parameters():
                p.requires_grad = True
            for  p in model.cnn.self_attention.parameters():
                p.requires_grad = True

            # for p in model.cnn.layer3.parameters():
            #     p.requires_grad = True
            # for p in model.cnn.layer2.parameters():
            #     p.requires_grad = True
            # for p in model.cnn.layer1.parameters():
            #     p.requires_grad = True
            # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
            # patch_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
            # word_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
            # model.fc  = nn.Linear(512, 1)#nn.Sequential(nn.Linear(512, 256),nn.Linear(256, 1))#3) #512

            model.to(device)
            
            # for name, param in model.named_parameters():
            #     # if param.requires_grad:
            #     print(name)
            # model.cnn.eval()
            # loss_fn = DynamicWeightedLoss(num_losses=3)

            # parameters = []
            # for name, param in model.named_parameters():
            #     if 'layer4' in name or 'fc' in name:
            #         parameters.append({'params': param, 'weight_decay': 0.001})
            #     else:
            #         parameters.append({'params': param, 'weight_decay': 0.0})
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)#,weight_decay=args.weight_decay)
            # print("pretrain",pretrain.keys())
            # optimizer.load_state_dict(pretrain['optimizer'])
            # optimizer = optim.Adam(parameters, lr=args.lr)
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            # scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=400,T_mult=1, eta_min=1e-8, last_epoch=-1)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            
            # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            #(optimizer, step_size=10, gamma=0.1)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            optimizer.zero_grad()
            cl_list=[]

            component_params, total_params = count_trainable_parameters(model.cnn)

            # # Print the results
            print("Components containing trainable parameters:")
            for component_name, num_params in component_params.items():
                print(f"{component_name}: {num_params} parameters")

            print(f"\nTotal number of trainable parameters: {total_params}")

            for epoch in range(args.num_epochs):
                model.train()
                # model.cnn.eval()
                # # model.eval()
                # model.cnn.layer4.train()
                # model.cnn.fc.train()
                # model.cnn.layer3.train()
                train_loss = 0
                counter = 0
                num_batches=0
                val_batch=0
                train_batches=0
                epoch_loss = 0
                training_ture=[]
                training_estimated=[]
                
                for num_batches,(images, text,labells,masks,location) in enumerate(train_dl): 
            
                    images ,labells= images.to(device),labells.to(device)
                    



        #            epoch_loss += total_loss.item()

                    
                    model.to(device)
            #       optimizer.zero_grad()
                    images=images.float()
                    pred,_= model(images)
                    prob = torch.sigmoid(pred)
                    # latent_space , _ = resnet(images.float())

                    # latent_space_tsne = tsne.fit_transform(latent_space.detach().cpu().numpy())
                    # plt.scatter(latent_space_tsne[:, 0], latent_space_tsne[:, 1])
                    # plt.title('t-SNE Visualization of Latent Space')
                    # plt.xlabel('t-SNE Dimension 1')
                    # plt.ylabel('t-SNE Dimension 2')
                    # os.chmod(f"/hpf/largeprojects/fkhalvati/Sara/MRI - x/tsne_bs1.png", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    # plt.savefig(f"/hpf/largeprojects/fkhalvati/Sara/MRI - x/tsne_bs1.png")
                    # print("pred",pred.shape,labells.shape)
                    loss=classifier_criterion(pred,labells.unsqueeze(1).float())  
                    
                    
                

                    loss.backward()#(retain_graph=True) # retain_graph=True done for tie net model
                    # -- clip the gradient at a specified value in the range of [-clip, clip].
                    # -- This is mainly used to prevent exploding or vanishing gradients in the network training
        #            nn.utils.clip_grad_value_(model.parameters(), 0.1)
        #         if num_batches==num_batch_accumulate: #If we have enough batches to take a step
                    if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                        optimizer.step() #after if 
                        optimizer.zero_grad() # after if
                    
                    for i in range(len(labells.tolist())):
                        training_ture.append(labells.tolist()[i])#[0])
                        training_estimated.append(prob.tolist()[i])#[0]) #prob
                
                    train_loss+=loss.item()*batch_size
                    #?
                    train_batches+=1
                    counter += 1
                # 230, 340, 410
                train_loss = train_loss/(train_batches)
                if epoch==19:
                    torch.save(model.state_dict(), os.path.join(args.output_dir,f"_ds_wl_cross{epoch}update_downstream_whole_MRI_classification__valid_fold__lr0.0003_{test_fold}{valid_fold}.pth"))
                scheduler.step()
                model.eval()
                # scheduler.step()
                with torch.set_grad_enabled(False):

                    val_loss = 0.0
                    val_b=0
                    total_epoch = 0
                    validation_true = []
                    validation_estimated = []
                    
                    for val_batches,(images,text,labells,masks,location) in enumerate(valid_dl):
                        
                        images, labells = images.to(device), labells.to(device)
                        
            
                        pred ,_ = model(images.float())
    
                        
                
                        
                        prob = torch.sigmoid(pred)
                        val_loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                        
                        for i in range(len(labells.tolist())):
                            validation_true.append(labells.tolist()[i])#[0])
                            validation_estimated.append(prob.tolist()[i])#[0])

                # scheduler.step(val_loss)   
                # print("herethere", validation_true)
                # print("herethere", validation_estimated)
                # print("herethere", labells,prob)
                val_auc = roc_auc_score(validation_true, validation_estimated)#,multi_class='ovr')
                
                train_auc = roc_auc_score(training_ture, training_estimated)#,multi_class='ovr')

                


            

                total_train_auc[epoch] = train_auc
                total_val_auc[epoch] = val_auc
                
                print("epoch",epoch,":","train_AUC:",train_auc,"val_AUC",val_auc)
                if epoch==args.num_epochs:
                    if val_auc>max_auc:
                        max_auc=val_auc
                        best_model=model

            model.eval()
            with torch.set_grad_enabled(False):

                
                test_true = []
                test_estimated = []
                
                for images,text,labells,masks,location in test_dl:
                    

                    images, labells = images.to(device), labells.to(device)
                    
            
                    pred ,_ = model(images.float())
                
                    prob = torch.sigmoid(pred)
                    for i in range(len(labells.tolist())):
                        test_true.append(labells.tolist()[i])#[0])
                        test_estimated.append(prob.tolist()[i])#[0])
                    # if epoch==args.num_epochs-1:
                    #     print("test AUC:", model_eval(model,test_dl))
            
                test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
                print("test_auc",test_auc)
                break
        # if test_fold==0:
                # break


    logging.info('Finished training.')
    return 0, total_train_auc, 0, total_val_auc



def train_downstream_image_text_cv(args, data_ds, output_model_path, tuning=False):
    
    

    print("here")
 #   optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
    batch_size=args.batch_size
    num_batch_accumulate = 16/batch_size #128 / batch_size

    classifier_criterion = nn.BCEWithLogitsLoss()#BCELoss()#.
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
#    optimizer.zero_grad()
    # weight_path = "/hpf/largeprojects/fkhalvati/Sara/pretrain/resnet_18_23dataset.pth"
    # weight_path=f"/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text/revised_attention_local_global__fold__0__epoch__{args.weight_epoch}__margin0.25"
    weight_path=f"/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text/revised_attention_local_global__fold__0__epoch__{args.weight_epoch}__margin0.25"
    best_auc=0
    outer_kfold = KFold(n_splits=5, shuffle=True,random_state=42)
    for test_fold, (outer_train_indices,outer_test_indices) in enumerate(outer_kfold.split(data_ds)):
        print(f"test_fold:{test_fold}")
        outer_train_set = torch.utils.data.Subset(data_ds, outer_train_indices)
        # outer_train_set = data_ds
        outer_test_set = torch.utils.data.Subset(data_ds, outer_test_indices)
    
        
        test_dl = DataLoader(outer_test_set, batch_size=batch_size)
        inner_kfold = KFold(n_splits=5, shuffle=True,random_state=42)

        # file = open('test_dataset_MRI', 'wb')

        # # # dump information to that file
        # pickle.dump(outer_test_set, file)

        # # # close the file
        # file.close()
        for valid_fold, (inner_train_indices, inner_val_indices) in enumerate(splits.split(outer_train_set)):

            print(f"valid_fold:{valid_fold}")

            inner_train_set = torch.utils.data.Subset(outer_train_set, inner_train_indices)
            inner_val_set = torch.utils.data.Subset(outer_train_set, inner_val_indices)

            

            # file = open('train_dataset_MRI', 'wb')

            # # # dump information to that file
            # pickle.dump(inner_train_set, file)

            # # # close the file
            # file.close()


            # file = open('valid_dataset_MRI', 'wb')

            # # # dump information to that file
            # pickle.dump(inner_val_set, file)

            # # # close the file
            # file.close()
            
            train_dl =DataLoader(inner_train_set, batch_size=batch_size, shuffle=True)        
            valid_dl = DataLoader(inner_val_set, batch_size=batch_size, shuffle=True)
            inplanes=[64, 128, 256, 512]
            # inplanes=[32,64, 128, 256]

            model=image_text()#downstream_image_text_classifier()
            model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
            # model.cnn.fc = nn.Linear(1039, 512)
            model.cnn.self_attention = SelfAttention(512)
            ##################3pretrain
            
            # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
            net_dict = model.cnn.state_dict()
            
            pretrain = torch.load(weight_path)
            # print("pretrain",pretrain)#["layer4.1.conv2.weight"])
            # pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
            # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items()}# if k in net_dict.keys()}#***
            
            # # print("pretrain",pretrain_dict)
            # net_dict.update(pretrain_dict)#***
            # print("net_dict,", net_dict["layer4.1.conv2.weight"])

            # del model.cnn.global_embed
            del model.cnn.local_embed
            pretrain_cnn={k: v for k, v in pretrain.items() if k.startswith("cnn") and "local_embed" not in k and "global_embed" not in k}# and "local_embed.conv1d_1" not in k and "local_embed.conv1d_2" not in k}# and "local_embed" not in k and "global_embed" not in k }#and "fc" not in k}
            pretrain_cnn = {k.replace('cnn.', ''): v for k, v in pretrain_cnn.items()}
            model.cnn.load_state_dict(pretrain_cnn)#***
            ##############3pretrain
            model.cnn.fc=nn.Linear(512, 256)
            model.cnn.fc1=nn.Linear(256,1)
            # model.cnn.fc=nn.Linear(320, 128)
            # model.cnn.fc1=nn.Linear(128,1)

            # model.cnn.self_attention = nn.MultiheadAttention(
            # embed_dim=512,num_heads=1, batch_first=True,dropout=0.5)
            # model.cnn.self_attention = SelfAttention(512)
            
            #rename

            # for name, module in model.named_children():
            #     print("module",name, module)     

            # model.load_state_dict(weight_path['state_dict'])  ####mednet
            # model.load_state_dict(pretrain) 
            # optimizer.load_state_dict(weight_path['optimizer']) ### mednet
            # for p in parameters:
            # del model.cnn.global_embed
            # del model.cnn.local_embed
            for pname, p in model.cnn.named_parameters():
                p.requires_grad = False
            for pname, p in model.cnn.fc.named_parameters():
                p.requires_grad = True
            for  p in model.cnn.layer4.parameters():
                p.requires_grad = True
            for  p in model.cnn.layer3.parameters():
                p.requires_grad = True
            for  p in model.cnn.fc1.parameters():
                p.requires_grad = True
            for  p in model.cnn.self_attention.parameters():
                p.requires_grad = True

            # for p in model.cnn.layer3.parameters():
            #     p.requires_grad = True
            # for p in model.cnn.layer2.parameters():
            #     p.requires_grad = True
            # for p in model.layer3.parameters():
            #     p.requires_grad = True
            # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
            # patch_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
            # word_local_atten_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
            # model.fc  = nn.Linear(512, 1)#nn.Sequential(nn.Linear(512, 256),nn.Linear(256, 1))#3) #512

            model.to(device)
            
            # model.cnn.eval()
            # loss_fn = DynamicWeightedLoss(num_losses=3)

            # parameters = []
            # for name, param in model.named_parameters():
            #     if 'layer4' in name or 'fc' in name:
            #         parameters.append({'params': param, 'weight_decay': 0.001})
            #     else:
            #         parameters.append({'params': param, 'weight_decay': 0.0})
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)#,weight_decay=args.weight_decay)
            # print("pretrain",pretrain.keys())
            # optimizer.load_state_dict(pretrain['optimizer'])
            # optimizer = optim.Adam(parameters, lr=args.lr)
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            # scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=400,T_mult=1, eta_min=1e-8, last_epoch=-1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)

            # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            #(optimizer, step_size=10, gamma=0.1)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            optimizer.zero_grad()
            cl_list=[]
            for epoch in range(args.num_epochs):
                model.train()
                # model.cnn.eval()
                # # model.eval()
                # model.cnn.layer4.train()
                # model.cnn.fc.train()
                # model.cnn.layer3.train()
                train_loss = 0
                counter = 0
                num_batches=0
                val_batch=0
                train_batches=0
                epoch_loss = 0
                training_ture=[]
                training_estimated=[]
    
                for num_batches,(images, text,labells,masks,location) in enumerate(train_dl): 
                    
                    images ,labells= images.to(device),labells.to(device)
                    
                    images = images.cuda()

                    mask = text['attention_mask'].to(device)
                    input_id = text['input_ids'].squeeze(1).to(device)
        #            epoch_loss += total_loss.item()

                    
                    model.to(device)
            #       optimizer.zero_grad()
                    images=images.float()
                    pred= model(images,input_id, mask)
                    prob = torch.sigmoid(pred)

                    # print("pred",pred.shape,labells.shape)
                    loss=classifier_criterion(pred,labells.unsqueeze(1).float())  
                    
                    
                

                    loss.backward()#(retain_graph=True) # retain_graph=True done for tie net model
                    # -- clip the gradient at a specified value in the range of [-clip, clip].
                    # -- This is mainly used to prevent exploding or vanishing gradients in the network training
        #            nn.utils.clip_grad_value_(model.parameters(), 0.1)
        #         if num_batches==num_batch_accumulate: #If we have enough batches to take a step
                    if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                        optimizer.step() #after if 
                        optimizer.zero_grad() # after if
                    
                    for i in range(len(labells.tolist())):
                        training_ture.append(labells.tolist()[i])#[0])
                        training_estimated.append(prob.tolist()[i])#[0]) #prob
                
                    train_loss+=loss.item()*batch_size
                    #?
                    train_batches+=1
                    counter += 1

                train_loss = train_loss/(train_batches)
                torch.save(model.state_dict(), os.path.join(args.output_dir,f"downstream_whole_MRI_report_{test_fold}{valid_fold}.pth"))
                # scheduler.step()
                model.eval()
                # scheduler.step()
                with torch.set_grad_enabled(False):

                    val_loss = 0.0
                    val_b=0
                    total_epoch = 0
                    validation_true = []
                    validation_estimated = []
                    
                    for val_batches,(images,text,labells,masks,location) in enumerate(valid_dl):
                        
                        mask = text['attention_mask'].to(device)
                        input_id = text['input_ids'].squeeze(1).to(device)
                        images,labells = images.to(device),labells.to(device)
                        
                        input_id,mask = input_id.cuda() , mask.cuda()
                        pred = model(images.float(),input_id, mask)
    
                        
                
                        
                        prob = torch.sigmoid(pred)
                        val_loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                        
                        for i in range(len(labells.tolist())):
                            validation_true.append(labells.tolist()[i])#[0])
                            validation_estimated.append(prob.tolist()[i])#[0])

                scheduler.step(val_loss)   
            
                val_auc = roc_auc_score(validation_true, validation_estimated)#,multi_class='ovr')
                
                train_auc = roc_auc_score(training_ture, training_estimated)#,multi_class='ovr')

                


            

                total_train_auc[epoch] = train_auc
                total_val_auc[epoch] = val_auc
                
                print("epoch",epoch,":","train_AUC:",train_auc,"val_AUC",val_auc)
                if epoch==args.num_epochs:
                    if val_auc>max_auc:
                        max_auc=val_auc
                        best_model=model

            model.eval()
            with torch.set_grad_enabled(False):

                
                test_true = []
                test_estimated = []
                
                for images,text,labells,masks,location in test_dl:
                    

                    images, labells = images.to(device),labells.to(device)
                    
                    mask = text['attention_mask'].to(device)
                    input_id = text['input_ids'].squeeze(1).to(device)
                    pred = model(images.float(),input_id,mask)
                
                    prob = torch.sigmoid(pred)
                    for i in range(len(labells.tolist())):
                        test_true.append(labells.tolist()[i])#[0])
                        test_estimated.append(prob.tolist()[i])#[0])
                    # if epoch==args.num_epochs-1:
                    #     print("test AUC:", model_eval(model,test_dl))
            
                test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
                print("test_auc",test_auc)
                # break
        # if test_fold==0:
        #     break


    logging.info('Finished training.')
    return 0, total_train_auc, 0, total_val_auc


def eval_downstream(model,test_dl,support_dl,mode):
    model = model.cuda()
    model.eval()
    threshold = 0.4
    tsne = TSNE(n_components=2, random_state=42)
    resnet = model.cnn
    with torch.set_grad_enabled(False):

        
        test_true = []
        test_estimated = []
        test_pred = []
        
        for images,_,labells,_,_  in test_dl:
            
            
            images, labells = images.to(device), labells.to(device)
            
            if mode == "downstream":
                pred , _= model(images.float())
                prob = torch.sigmoid(pred)
                predicted_labels = torch.where(prob >= threshold, torch.tensor(1,device="cuda"), torch.tensor(0,device="cuda"))

                # latent_space , _ = resnet(images.float())
                # latent_space_tsne = tsne.fit_transform(latent_space)
                # plt.scatter(latent_space_tsne[:, 0], latent_space_tsne[:, 1])
                # plt.title('t-SNE Visualization of Latent Space')
                # plt.xlabel('t-SNE Dimension 1')
                # plt.ylabel('t-SNE Dimension 2')
                # plt.savefig(f"/hpf/largeprojects/fkhalvati/Sara/MRI - x/tsne_bs1.png")
            elif mode== "zero_shot":
                label_list = [0,1]
                pred = zeroshot_classifier(label_list, images , model)
                print("preddd",pred)
                prob = pred
            elif mode== "few_shot":
                
                pred = fewshot_classifier(images, support_dl , model)
                print("preddd",pred)
                prob = pred
            # test_true.append(labells) #####################3
            # # print("probbbb",prob)
            # # print("probbb",torch.Tensor([prob]))
            # test_estimated.append(torch.Tensor([prob])) ###############3333
            # print(torch.Tensor(prob).tolist())
            for i in range(len(labells.tolist())):
                # print("lennn",len(torch.Tensor(prob).tolist()))
                # print("labells",labells,prob)
                test_true.append(labells.tolist()[i])
                test_estimated.append(prob.tolist()[i][0])
                test_pred.append(predicted_labels.tolist()[i][0])
            # if epoch==args.num_epochs-1:
            #     print("test AUC:", model_eval(model,test_dl))
        # print("typeeeeee",type(test_true),type(test_estimated))
        print("lennnn",test_true,test_estimated,test_pred)
        test_auc = roc_auc_score(torch.Tensor(test_true), torch.Tensor(test_estimated))#,multi_class='ovr')
        test_precision = precision_score(torch.Tensor(test_true), torch.Tensor(test_pred))
        test_recall = recall_score(torch.Tensor(test_true), torch.Tensor(test_pred))
        test_f1_score = f1_score(torch.Tensor(test_true), torch.Tensor(test_pred))
        print("test_auc",test_auc)
        print("test_precision",test_precision)
        print("test_recall",test_recall)
        print("test_f1-score",test_f1_score)
        return test_auc,test_precision,test_recall,test_f1_score
        


def eval_downstream_image_text(model,test_dl,support_dl,mode="downstream"):
    model = model.cuda()
    model.eval()
    threshold = 0.4
    with torch.set_grad_enabled(False):

        
        test_true = []
        test_estimated = []
        test_pred = []
        
        for images,texts, labells,_,_  in test_dl:
            
            
            images, labells = images.cuda(), labells.cuda()
            
            mask = texts['attention_mask'].cuda()
            input_id = texts['input_ids'].squeeze(1).cuda()
            if mode == "downstream":
                pred , _= model(images.float(),input_id,mask)
                prob = torch.sigmoid(pred)
                predicted_labels = torch.where(prob >= threshold, torch.tensor(1,device="cuda"), torch.tensor(0,device="cuda"))
            elif mode== "zero_shot":
                label_list = [0,1]
                pred = zeroshot_classifier(label_list, images , model)
                # print("preddd",pred)
                prob = pred
            # test_true.append(labells) #####################3
            # # print("probbbb",prob)
            # # print("probbb",torch.Tensor([prob]))
            # test_estimated.append(torch.Tensor([prob])) ###############3333
            # print(torch.Tensor(prob).tolist())
            for i in range(len(labells.tolist())):
                # print("lennn",len(torch.Tensor(prob).tolist()))
                # print("labells",labells,prob)
                test_true.append(labells.tolist()[i])#[0])
                test_estimated.append(prob.tolist()[i][0])
                test_pred.append(predicted_labels.tolist()[i][0])
            # if epoch==args.num_epochs-1:
            #     print("test AUC:", model_eval(model,test_dl))
        # print("typeeeeee",type(test_true),type(test_estimated))
        test_auc = roc_auc_score(torch.Tensor(test_true), torch.Tensor(test_estimated))#,multi_class='ovr')
        test_precision = precision_score(test_true, test_pred)
        test_recall = recall_score(test_true, test_pred)
        test_f1_score = f1_score(test_true, test_pred)
        print("test_auc",test_auc)
        print("test_precision",test_precision)
        print("test_recall",test_recall)
        print("test_f1-score",test_f1_score)
        return test_auc,test_precision,test_recall,test_f1_score

def eval_downstream_text(model,test_dl,support_dl,mode="downstream"):
    model = model.cuda()
    model.eval()
    threshold = 0.4
    with torch.set_grad_enabled(False):

        
        test_true = []
        test_estimated = []
        test_pred = []
        
        for images,texts, labells,_,_  in test_dl:
            
            
            labells = labells.cuda()
            
            mask = texts['attention_mask'].cuda()
            input_id = texts['input_ids'].squeeze(1).cuda()
            if mode == "downstream":
                pred = model(input_id,mask)
                prob = torch.sigmoid(pred)
                predicted_labels = torch.where(prob >= threshold, torch.tensor(1,device="cuda"), torch.tensor(0,device="cuda"))

            elif mode== "zero_shot":
                label_list = [0,1]
                pred = zeroshot_classifier(label_list, images , model)
                # print("preddd",pred)
                prob = pred
            # test_true.append(labells) #####################3
            # # print("probbbb",prob)
            # # print("probbb",torch.Tensor([prob]))
            # test_estimated.append(torch.Tensor([prob])) ###############3333
            # print(torch.Tensor(prob).tolist())
            for i in range(len(labells.tolist())):
                # print("lennn",len(torch.Tensor(prob).tolist()))
                # print("labells",labells,prob)
                test_true.append(labells.tolist()[i])#[0])
                test_estimated.append(prob.tolist()[i][0])
                test_pred.append(predicted_labels.tolist()[i][0])
            # if epoch==args.num_epochs-1:
            #     print("test AUC:", model_eval(model,test_dl))
        # print("typeeeeee",type(test_true),type(test_estimated))
        test_auc = roc_auc_score(torch.Tensor(test_true), torch.Tensor(test_estimated))#,multi_class='ovr')
        test_precision = precision_score(test_true, test_pred)
        test_recall = recall_score(test_true, test_pred)
        test_f1_score = f1_score(test_true, test_pred)
        print("test_auc",test_auc)
        print("test_precision",test_precision)
        print("test_recall",test_recall)
        print("test_f1-score",test_f1_score)
        return test_auc,test_precision,test_recall,test_f1_score



if __name__ == '__main__':
    print("ds classification _ wl models")
    args = make_parser().parse_args()

#    args.device = torch.device("cuda:"+ args.gpus) if cuda else torch.device('cpu')

    image_folder=args.image_path###################################################################################3
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    file = open('input_tokenized_text', 'rb') #important
    
# dump information to that file

    df_text_excel = pickle.load(file)

# close the file
    file.close()
    data=process_excel(df_text_excel)
    data.index=range(data.shape[0])
    
    

    

    df_loc = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Nomogram_study_LGG_data_Nov.27.xlsx",engine='openpyxl')
    
    dataset2 = BertDataset(data,image_folder,df_loc)
    #######3
    test_image_folder = os.path.join("/hpf/largeprojects/fkhalvati/Datasets/MedicalImages/BrainData/SickKids/preprocessed_pLGG_EN_Nov2023_KK")
    df_loc = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Nomogram_study_LGG_data_Nov.27.xlsx",sheet_name="Stanford",engine='openpyxl')
    
    # df = pd.read_csv("/hpf/largeprojects/fkhalvati/Sara/pLGG_4cohorts_532subs.csv")#,engine='openpyxl')
    
    # df = df[df["folder_name"].notnull()]
    
    
    
    # dataset2 = Eval_new_sk(df,test_image_folder)
     
    file = open('train_dataset2', 'rb') #important
    
# dump information to that file

    dataset = pickle.load(file)

# close the file
    file.close()
    
    
    # train_dataset,test_dataset = split_dataset_cv(dataset,0.8)#,0.6,0.2)#,0.6,0.2)
    # file = open('test_dataset2', 'wb')

    # # dump information to that file
    # pickle.dump(test_dataset, file)

    # # close the file
    # file.close()
    # test_dl=load_data(test_dataset,test_dataset,test_dataset, args.batch_size)[2]
    inplanes=[64, 128, 256, 512]#512*2]
    if args.stage=="training":
        # test_image_folder = os.path.join("/hpf/largeprojects/fkhalvati/Datasets/MedicalImages/BrainData/SickKids/preprocessed_pLGG_EN_Nov2023_KK")
        # df = pd.read_csv("/hpf/largeprojects/fkhalvati/Sara/pLGG_4cohorts_532subs.csv")
        # df = df[df["folder_name"].notnull()]
        # test_dataset = Eval_new_sk(df,test_image_folder)
        train_lossss, train_aucsss, val_lossss, val_aucsss=train_downstream_image_model_cv(args,dataset2,args.output_dir)#(args,train_dataset,test_dl,args.output_dir)
        # train_lossss, train_aucsss, val_lossss, val_aucsss=train_downstream_image_text_cv(args,dataset2,args.output_dir)#(args,train_dataset,test_dl,args.output_dir)
        # train_lossss, train_aucsss, val_lossss, val_aucsss=train_image_model_cv(args,dataset2,args.output_dir)#(args,train_dataset,test_dl,args.output_dir)


    elif args.stage=="zero_shot":
        weight_path=f"/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text/_na_location_attention_local_global__fold__0__epoch__{args.weight_epoch}__margin0.25"
        model=image_text_attention(emb_dim=128,num_heads=2,mode="global_local") 
        model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)

        model.cnn.self_attention = SelfAttention(512)
        model.cnn.local_embed = LocalEmbedding_3d(
            256, 256,256
        )
        # print("local",model.cnn.local_embed.head)
        net_dict = model.state_dict()
        
        pretrain = torch.load(weight_path)
        # print("pretrain1",[i for i in net_dict.keys()])# if "cnn.local_embed" in i])
        # print("pretrain",[i for i in pretrain.keys() if "cnn.local_embed" in i])
        # print("keyyyyyss", pretrain.keys())
        # pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
        # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        pretrain_dict = {k: v for k, v in pretrain.items()}#if k in net_dict.keys()}
        # net_dict.update(pretrain_dict)
        model.load_state_dict(pretrain_dict)

        # test_image_folder = os.path.join("/hpf/largeprojects/fkhalvati/Sara/new_sk2_preprocessing/")
        test_image_folder = os.path.join("/hpf/largeprojects/fkhalvati/Datasets/MedicalImages/BrainData/SickKids/preprocessed_pLGG_EN_Nov2023_KK")

        df = pd.read_csv("/hpf/largeprojects/fkhalvati/Sara/pLGG_4cohorts_532subs.csv")
        df = df[df["folder_name"].notnull()]
        name_list = os.listdir(test_image_folder)
        df = df[df["folder_name"].isin(name_list)]
        df.index = range(len(df))
        test_dataset = Eval_new_sk(df,test_image_folder)
        test_dl = DataLoader(test_dataset, batch_size=1)#args.batch_size)
        test_auc =  eval_downstream(model,test_dl, test_dl , mode="zero_shot") 
        print("test_auc",test_auc)
        
    elif args.stage=="few_shot":
        weight_path=f"/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text/_local_if_updated_location_attention_local_global__fold__0__epoch__{args.weight_epoch}__margin0.25"
        model=image_text_attention(emb_dim=128,num_heads=2,mode="global_local") 
        model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)

        model.cnn.self_attention = SelfAttention(512)
        model.cnn.local_embed = LocalEmbedding_3d(
            256, 256,256
        )
        # print("local",model.cnn.local_embed.head)
        res = model.cnn
        net_dict = model.state_dict()
        
        pretrain = torch.load(weight_path)
        # print("pretrain1",[i for i in net_dict.keys()])# if "cnn.local_embed" in i])
        # print("pretrain",[i for i in pretrain.keys() if "cnn.local_embed" in i])
        # print("keyyyyyss", pretrain.keys())
        # pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
        # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        pretrain_dict = {k: v for k, v in pretrain.items()}#if k in net_dict.keys()}
        # net_dict.update(pretrain_dict)
        model.load_state_dict(pretrain_dict)

        # test_image_folder = os.path.join("/hpf/largeprojects/fkhalvati/Sara/new_sk2_preprocessing/")
        test_image_folder = os.path.join("/hpf/largeprojects/fkhalvati/Datasets/MedicalImages/BrainData/SickKids/preprocessed_pLGG_EN_Nov2023_KK")

        df = pd.read_csv("/hpf/largeprojects/fkhalvati/Sara/pLGG_4cohorts_532subs.csv")
        df = df[df["folder_name"].notnull()]
        name_list = os.listdir(test_image_folder)
        df = df[df["folder_name"].isin(name_list)]
        df.index = range(len(df))
        print("3333333333",df.index)
        test_dataset = Eval_new_sk(df,test_image_folder)
        support_dataset = fewshot_support(df,test_image_folder)
        test_dl = DataLoader(test_dataset, batch_size=1)#args.batch_size)
        support_dl = DataLoader(support_dataset, batch_size=1)
        test_auc =  eval_downstream(model,test_dl,support_dl , mode="few_shot") 
        print("test_auc",test_auc)
        
    else:  
        # with torch.set_grad_enabled(False):
        image_folder=args.image_path   
    
   
    
        file = open('input_tokenized_text', 'rb') #important
        df_text_excel = pickle.load(file)
        file.close()
        data=process_excel(df_text_excel)
        data.index=range(data.shape[0])
      
        df_loc_1 = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Nomogram_study_LGG_data_Nov.27.xlsx",engine='openpyxl')
        #### model initialization
        model=downstream_image_classifier()
        model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        model.cnn.self_attention = SelfAttention(512)
        
        model.cnn.fc=nn.Linear(512, 256)
        model.cnn.fc1=nn.Linear(256,1)
        #### model initialization
        # model=image_text()
        # model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        # model.cnn.self_attention = SelfAttention(512)
        #### model initialization
        # model = BertClassifier()
        # model.fc1=nn.Linear(1024, 512)
        # model.fc3=nn.Linear(512,256)
        # # self.fc1=nn.Linear(524288, 1024)
        # # self.fc4=nn.Linear(256,128)
        # model.fc2=nn.Linear(256, 1)
        #### model initialization

        # del model.cnn.global_embed
        # del model.cnn.local_embed
        auc_list=[]
        precision_list=[]
        recall_list = []
        f1_list = []
        # for i,path in enumerate([2,1,4,0,4]):
        # test_image_folder = os.path.join("/hpf/largeprojects/fkhalvati/Sara/new_sk2_preprocessing/")
        test_image_folder = os.path.join("/hpf/largeprojects/fkhalvati/Datasets/MedicalImages/BrainData/SickKids/preprocessed_pLGG_EN_Nov2023_KK")
        df_loc = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Nomogram_study_LGG_data_Nov.27.xlsx",sheet_name="Stanford",engine='openpyxl')
        # df2 = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Stanford_new_data_09_21.xlsx",engine='openpyxl')
        df = pd.read_csv("/hpf/largeprojects/fkhalvati/Sara/pLGG_4cohorts_532subs.csv")#,engine='openpyxl')
        # df2=df2.iloc[:10]
        df = df[df["folder_name"].notnull()]
        # print("columnsss",df2.loc[9,'Unnamed: 0'])
        

        # dataset = Eval_new_sk(data,image_folder)
#         file = open('test_dataset2', 'rb') #important
    
# # dump information to that file

#         test_dataset = pickle.load(file)

        
        
        # df = df[df["code"].notnull()]
        
        
        
        test_dataset = Eval_new_sk(df,test_image_folder)


        # test_dataset = BertDataset(data,image_folder,df_loc_1)
        # test_dl = DataLoader(test_dataset, batch_size=args.batch_size)
        # test_auc =  eval_downstream(model,test_dl)   
        # auc_list.append(test_auc)
        # print("new auc_list",auc_list)
        # print("new mean test auc:", np.mean(auc_list))    
        # for i,path in enumerate ([2,1,2,4,4]):
        # for i,path in enumerate([4,1,0,4,4]):
        # for i,path in enumerate([4,1,2,4,4]):
        for i,path in enumerate([0,0,0,0,0]):
        
            # print("i_path",i,path)
            
            # if i<=1 or i==4:
            #     continue
            # weight_path = os.path.join(args.output_dir,f"790_cross9update_downstream_whole_MRI_classification__valid_fold__lr0.0003_{i}{path}.pth") MRI_m3d
            # weight_path = os.path.join(args.output_dir,f"ROI_base_MRI_classification__resnet18_23_valid_fold__{i}{path}.pth")
            # weight_path = os.path.join(args.output_dir,f"whole_MRI_withoutlayer3_classification__resnet18_23_valid_fold__{i}{path}.pth")
            # weight_path = os.path.join(args.output_dir,f"ROI_image_text_tl_attention_fold__{i}{path}")
            # weight_path = os.path.join(args.output_dir,f"image_text_tl_attention_fold__{i}{path}")
            # weight_path = os.path.join(args.output_dir,f"new_image_text_tl_attention_fold__{i}{path}")
            # weight_path = os.path.join(args.output_dir,f"transformer_baseline_text_lr(9e-5)_({i+1}, {path+1}).pth")#{i+1}{path+1}")
            # weight_path = os.path.join(args.output_dir,f"whole_MRI_withoutlayer3_classification__valid_fold__lr0.0001_0{path}.pth")
            weight_path=f"/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text/_ds_local_cross19update_downstream_whole_MRI_classification__valid_fold__lr0.0003_00.pth"

            # if i==0:
            #     weight_path = os.path.join(args.output_dir,"epochs",f"update_downstream_whole_MRI_classification__valid_fold__lr0.0003_{i}{0}.pth")
            # else:
            #     weight_path = os.path.join(args.output_dir,"epochs",f"la4update_downstream_whole_MRI_classification__valid_fold__lr0.0003_{i}0.pth")
            # weight_path =   os.path.join(args.output_dir,f"_scratch_depth_base_MRI_classification__resnet18_23_valid_fold__{i}{path}.pth")
            ####
            loaded={k: v for k, v in torch.load(weight_path).items() }#if k in model.state_dict().keys()}
            # print("loaded",loaded)
            # model.load_state_dict(torch.load(weight_path))
            model.load_state_dict(loaded)
            
            model.eval()
            # df = df[df["code"].notnull()]
            
            # print("df",df.shape)
            
            # test_dataset = EvalDataset(df1,df2,test_image_folder)
            # file = open(f'test_dataset_MRI{i}', 'rb') #important
            # test_dataset = pickle.load(file)
            # file.close()
            test_dl = DataLoader(test_dataset, batch_size=args.batch_size)
            test_auc, test_precision, test_recall, test_f1 =  eval_downstream(model,test_dl,test_dl , mode = "downstream")   
            auc_list.append(test_auc) #,test_precision,test_recall,test_f1_score
            precision_list.append(test_precision)
            recall_list.append(test_recall)
            f1_list.append(test_f1)


        # print("auc_list",auc_list)
        print("mean test auc:", np.mean(auc_list),np.std(auc_list))  
        
        print("mean test precision:", np.mean(precision_list),np.std(precision_list))
        
        print("mean test recall:", np.mean(recall_list),np.std(recall_list))
        
        print("mean test f1-score:", np.mean(f1_list),np.std(f1_list))       











