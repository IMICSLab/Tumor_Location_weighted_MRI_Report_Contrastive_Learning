# baseline MRI-based models
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
from data.fused_dataset import split_dataset_cv,process_excel,BertDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from data.fused_dataset import BertDataset,split_dataset_cv,process_excel,EvalDataset,Eval_new_sk, fewshot_support
# from fused_model_copy import generate_model
from model.fused_model import ResNet_attention,downstream_image_classifier,SelfAttention,image_text_attention, image_text, DepthAttention, MultiHeadSelfAttention, MultiHeadDepthAttention, LocalEmbedding_3d, BertClassifier
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

    
    parser.add_argument('--data_path', type=str, default="sth")
    parser.add_argument('--image_path', type=str, default="sth)
    parser.add_argument('--output_dir', type=str, default='sth/pLGG_results/image_text')
   
    
    parser.add_argument('--batch_size', type=int, default=4)  
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate') 
    

    
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--rseed', type=int, default=42, help='random seed')
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



        
def train_downstream_image_model_cv(args, data_ds, output_model_path, tuning=False):
    
    

    print("here")
    

    batch_size=args.batch_size
    num_batch_accumulate = 16/batch_size #128 / batch_size

    classifier_criterion = nn.BCEWithLogitsLoss()
    total_train_auc={}
    total_val_auc={}
    test_auc=[]

    weight_path = f"sth/pLGG_results/image_text/_without_location_if_updated_location_attention_local_global__fold__0__epoch__340__margin0.25"
   
    best_auc=0
    outer_kfold = KFold(n_splits=5, shuffle=True,random_state=42)
    for test_fold, (outer_train_indices,outer_test_indices) in enumerate(outer_kfold.split(data_ds)):
        
        print(f"test_fold:{test_fold}")
        print("newwwwwwwwwwwwwwww")
        outer_train_set = torch.utils.data.Subset(data_ds, outer_train_indices)
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
            

            model=downstream_image_classifier()
            model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
            model.cnn.self_attention = SelfAttention(512)
            ##################3pretrain
            
            
            # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
            net_dict = model.cnn.state_dict()
            
            pretrain = torch.load(weight_path)
            
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

        
            model.to(device)
            
           
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)#,weight_decay=args.weight_decay)
            # print("pretrain",pretrain.keys())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            optimizer.zero_grad()
            cl_list=[]

            component_params, total_params = count_trainable_parameters(model.cnn)

            print("Components containing trainable parameters:")
            for component_name, num_params in component_params.items():
                print(f"{component_name}: {num_params} parameters")

            print(f"\nTotal number of trainable parameters: {total_params}")

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
                
                for num_batches,(images, text,labells,masks,location) in enumerate(train_dl): 
            
                    images ,labells= images.to(device),labells.to(device)
                    



     

                    
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
                    
                    loss=classifier_criterion(pred,labells.unsqueeze(1).float())  
                    
                    loss.backward()
                    
                    if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):  #
                        optimizer.step()  
                        optimizer.zero_grad() 
                    
                    for i in range(len(labells.tolist())):
                        training_ture.append(labells.tolist()[i])
                        training_estimated.append(prob.tolist()[i])
                
                    train_loss+=loss.item()*batch_size
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
                   
            
                test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
                print("test_auc",test_auc)
                break
        # if test_fold==0:
                # break


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
                
            for i in range(len(labells.tolist())):
                # print("lennn",len(torch.Tensor(prob).tolist()))
                # print("labells",labells,prob)
                test_true.append(labells.tolist()[i])
                test_estimated.append(prob.tolist()[i][0])
                test_pred.append(predicted_labels.tolist()[i][0])
           
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
        

if __name__ == '__main__':
    print("ds classification _ wl models")
    args = make_parser().parse_args()

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
    
    

    

    df_loc = pd.read_excel("sth.xlsx",engine='openpyxl')
    
    dataset2 = BertDataset(data,image_folder,df_loc)
    #######3
    test_image_folder = os.path.join("sth")
    df_loc = pd.read_excel("sth.xlsx",sheet_name="sth",engine='openpyxl')

    # dataset2 = Eval_new_sk(df,test_image_folder)
     
    file = open('train_dataset2', 'rb') #important
    
# dump information to that file

    dataset = pickle.load(file)

# close the file
    file.close()
    
    
   
    inplanes=[64, 128, 256, 512]#512*2]
    if args.stage=="training":
       
        train_lossss, train_aucsss, val_lossss, val_aucsss=train_downstream_image_model_cv(args,dataset2,args.output_dir)
        
 
    else:  
        
        image_folder=args.image_path   
    
        file = open('input_tokenized_text', 'rb') #important
        df_text_excel = pickle.load(file)
        file.close()
        data=process_excel(df_text_excel)
        data.index=range(data.shape[0])
      
        df_loc_1 = pd.read_excel("sth.xlsx",engine='openpyxl')
        
        model=downstream_image_classifier()
        model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        model.cnn.self_attention = SelfAttention(512)
        
        model.cnn.fc=nn.Linear(512, 256)
        model.cnn.fc1=nn.Linear(256,1)
        
       
        auc_list=[]
        precision_list=[]
        recall_list = []
        f1_list = []
        
        test_image_folder = os.path.join("sth")
        df_loc = pd.read_excel("sth.xlsx",sheet_name="sth",engine='openpyxl')
        
        df = pd.read_csv("sth.csv")
        
        df = df[df["folder_name"].notnull()]
        
    
        
        test_dataset = Eval_new_sk(df,test_image_folder)


        for i,path in enumerate([0,0,0,0,0]):
        
           
            # weight_path = os.path.join(args.output_dir,f"790_cross9update_downstream_whole_MRI_classification__valid_fold__lr0.0003_{i}{path}.pth") MRI_m3d
            # weight_path = os.path.join(args.output_dir,f"ROI_base_MRI_classification__resnet18_23_valid_fold__{i}{path}.pth")
            # weight_path = os.path.join(args.output_dir,f"whole_MRI_withoutlayer3_classification__resnet18_23_valid_fold__{i}{path}.pth")
            # weight_path = os.path.join(args.output_dir,f"ROI_image_text_tl_attention_fold__{i}{path}")
            # weight_path = os.path.join(args.output_dir,f"image_text_tl_attention_fold__{i}{path}")
            # weight_path = os.path.join(args.output_dir,f"new_image_text_tl_attention_fold__{i}{path}")
            # weight_path = os.path.join(args.output_dir,f"transformer_baseline_text_lr(9e-5)_({i+1}, {path+1}).pth")#{i+1}{path+1}")
            # weight_path = os.path.join(args.output_dir,f"whole_MRI_withoutlayer3_classification__valid_fold__lr0.0001_0{path}.pth")
            weight_path=f"sth/_ds_local_cross19update_downstream_whole_MRI_classification__valid_fold__lr0.0003_{i}{path}.pth"

           
            ####
            loaded={k: v for k, v in torch.load(weight_path).items() }
            # print("loaded",loaded)
            
            model.load_state_dict(loaded)
            
            model.eval()
           
            
            test_dl = DataLoader(test_dataset, batch_size=args.batch_size)
            test_auc, test_precision, test_recall, test_f1 =  eval_downstream(model,test_dl,test_dl , mode = "downstream")   
            auc_list.append(test_auc)
            precision_list.append(test_precision)
            recall_list.append(test_recall)
            f1_list.append(test_f1)


        # print("auc_list",auc_list)
        print("mean test auc:", np.mean(auc_list),np.std(auc_list))  
        
        print("mean test precision:", np.mean(precision_list),np.std(precision_list))
        
        print("mean test recall:", np.mean(recall_list),np.std(recall_list))
        
        print("mean test f1-score:", np.mean(f1_list),np.std(f1_list))       











