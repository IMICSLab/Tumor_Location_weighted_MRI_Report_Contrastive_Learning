import torch
import pandas as pd
from captum.attr import LayerGradCam,GuidedGradCam,Lime
import matplotlib.pyplot as plt
from model.fused_model import downstream_image_classifier,SelfAttention,DepthAttention
from captum.attr import IntegratedGradients,LRP,FeatureAblation,ShapleyValues,FeaturePermutation,Saliency,Occlusion
import torch.nn as nn
import numpy as np
from PIL import Image
import os 
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from data.fused_dataset import split_dataset_cv,process_excel,BertDataset,Eval_new_sk
import pickle
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from model.fused_model import image_text,ResNet_attention,BasicBlock, ResNet_attention2
import nibabel as nib
from PIL import Image
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import cv2
import stat
from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from skimage.metrics import contingency_table
from medcam import medcam
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes




def soft_dice_coefficient(y_true, y_pred, smooth):  # extracted from: https://github.com/myidispg/kaggle-cloud/blob/20f20a9be25d872e5c5d5ec62b28ad656d516270/utils/helpers.py#L61
    # print("y_pred",y_true.shape,y_pred.shape)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    intersection = torch.sum(y_true_flat * y_pred_flat)
    union = torch.sum(y_true_flat) + torch.sum(y_pred_flat)
    dice_coeff = (2. * intersection + smooth) / (union + smooth)
    return dice_coeff

def dice_coefficient(y_pred, y_true, smooth=1.0):
    # print("preeeeeeeeeeee",predicted.shape,target.shape)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    intersection = torch.sum(y_pred_flat * y_true_flat)
    union = torch.sum(y_pred_flat) + torch.sum(y_true_flat) + smooth
    dice = (2.0 * intersection + smooth) / union
    return dice





def normalize_heatmap(heatmap):
    min_val = torch.min(heatmap)
    max_val = torch.max(heatmap)
    normalized_heatmap = (heatmap - min_val) / (max_val - min_val)
    return normalized_heatmap





inplanes=[64, 128, 256, 512]

weight_path = "sth/pLGG_results/image_text"
weight_path1 = "sth/pLGG_results"




# # 
#####CHANGE
model=downstream_image_classifier()
model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
model.cnn.fc=nn.Linear(512, 256)
model.cnn.fc1=nn.Linear(256,1)
model.cnn.self_attention=SelfAttention(512)
# model.cnn.self_attention2=SelfAttention()
# model.cnn.depth_attention=DepthAttention(512)




# model=image_text()
        
# inplanes=[64, 128, 256, 512]
# model.cnn=ResNet_attention2(True,BasicBlock,[2,2,2,2],model_depth=18, n_classes=1039,block_inplanes=inplanes,output_dim=256,mode="downstream")	#(BasicBlock,[2,2,2,2],model_depth=18, n_classes=1039,block_inplanes=inplanes,output_dim=256)
# model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2 , 2, 2), padding=(3, 3, 3), bias=False)
# model.cnn.self_attention=SelfAttention(512)
# model.cnn.depth_attention=DepthAttention(512)

# del model.cnn.local_embed
# del model.cnn.global_embed
# net_dict = model.cnn.state_dict()
model = model.cuda()
# model.eval()




# file = open('train_dataset_MRI', 'rb') #important
    
# # dump information to that files

# train_dataset = pickle.load(file)

# # close the file
# file.close()

# file = open('valid_dataset_MRI', 'rb') #important
    
# # dump information to that file

# valid_dataset = pickle.load(file)

# # close the file
# file.close()

# file = open('test_dataset_MRI', 'rb') #important
    
# # dump information to that file

# test_dataset = pickle.load(file)

# # close the file
# file.close()


# test_dl = DataLoader(test_dataset, batch_size=1)
# train_dl =DataLoader(train_dataset, batch_size=1, shuffle=True)        
# valid_dl = DataLoader(valid_dataset, batch_size=1, shuffle=True)





    


def main_slice(batch_masks,batch_heatmaps,batch_MRI):
    # slice_numbers = []
    batch_mask_list = []
    batch_heatmap_list = []
    batch_MRI_list = []
    # print("batchhhh",batch_masks.shape)
    for j,masks in enumerate(batch_masks):
    # Convert SimpleITK image to NumPy array for computation
        cross_section_areas = []
        
        for i in range(masks.shape[3]):
            mask = masks [:,:,:,i]
            np_mask = np.array(mask.cpu())
            tumor_area = np.sum(np_mask)  # Calculate the area of the tumor in this slice
            cross_section_areas.append(tumor_area)
        # slice_numbers.append(np.argmax(cross_section_areas))
        batch_mask_list.append(batch_masks[j,:,:,:,np.argmax(cross_section_areas)])
        batch_heatmap_list.append(batch_heatmaps[j,:,:,:,np.argmax(cross_section_areas)])
        batch_MRI_list.append(batch_MRI[j,:,:,:,np.argmax(cross_section_areas)])
     
    return torch.stack(batch_mask_list, dim=0).squeeze(-1), torch.stack(batch_heatmap_list, dim=0).squeeze(-1),torch.stack(batch_MRI_list, dim=0).squeeze(-1)





sigmoid_func = nn.Sigmoid()
v_list = [0,0,0,0,0]
IOU_list=[]
slice_list=[]
distance_list = []
with torch.no_grad():
    for i in range(5):
        #CHANGE
        # pretrain = torch.load(os.path.join(weight_path,f"downstream_whole_MRI_classification__valid_fold__lr0.0003_{0}{0}.pth" ))
        # pretrain = torch.load(os.path.join(weight_path,f"whole_MRI_withoutlayer3_classification__resnet18_23_valid_fold__{i}{v_list[i]}.pth" ))
        # pretrain = torch.load(os.path.join(weight_path,f"test_whole_MRI_withoutlayer3_classification__valid_fold__lr0.0001_{i}{v_list[i]}.pth" ))
        # pretrain = torch.load(os.path.join(weight_path,f"_scratch_depth_base_MRI_classification__resnet18_23_valid_fold__{i}{v_list[i]}.pth"))
        pretrain = torch.load(os.path.join(weight_path,f"790_cross19update_downstream_whole_MRI_classification__valid_fold__lr0.0003_{i}{v_list[i]}.pth"))
        # pretrain = torch.load(os.path.join(weight_path1,f"new_image_text_tl_attention_fold__{i}{v_list[i]}" ))

        # print("pretrain",pretrain.keys())
        pretrain_cnn={k: v for k, v in pretrain.items()  if "local_embed" not in k and k!="conv1.weight"}# and "cnn.self_attention.bn1" not in k}#if k.startswith("cnn") and "local_embed" not in k}# and "intermediate_layer" not in k}# and "local_embed" not in k and "global_embed" not in k}# and "local_embed.conv1d_1" not in k and "local_embed.conv1d_2" not in k }#and "local_embed" not in k and "global_embed" not in k }#and "fc" not in k}
        # pretrain_cnn = {k.replace('cnn.', ''): v for k, v in pretrain_cnn.items()}
        # print("model dict", {k for k in pretrain.keys()  if "local_embed" not in k})
        # print("tl dict", model.state_dict)
        # print("keyyyyyyyyyy",pretrain_cnn.keys())
        model.load_state_dict(pretrain_cnn)#***
        # model.load_state_dict(pretrain)#***
        
        
        # model.cuda()
        
        # del model.cnn.global_embed
        # del model.cnn.local_embed
        model.eval()

        
        
        
        # file = open(os.path.join(split_path,f'train_dataset_MRI2({i}, {v_list[i]})'), 'rb') #important
        file = open(f'train_dataset_MRI({i}, {0})', 'rb')
        # dump information to that file

        train_dataset = pickle.load(file)

        # close the file
        file.close()

        # file = open(os.path.join(split_path,f'valid_dataset_MRI2({i}, {v_list[i]})'), 'rb') #important
        file = open(f'valid_dataset_MRI({i}, {0})', 'rb')  
        # dump information to that file

        valid_dataset = pickle.load(file)

        # close the file
        file.close()

        # file = open(os.path.join(split_path,f'test_dataset_MRI2{i}'), 'rb') #important
        file = open(f'test_dataset_MRI{i}', 'rb') #important

        # dump information to that file

        test_dataset = pickle.load(file)

        # close the file
        file.close()
        #CHANGE
        # test_image_folder = os.path.join("sth")
        # df = pd.read_csv("sth')
        # df = df[df["folder_name"].notnull()]
        # test_dataset = Eval_new_sk(df,test_image_folder)

        test_dl = DataLoader(test_dataset, batch_size=8)
        train_dl =DataLoader(train_dataset, batch_size=8, shuffle=True)        
        valid_dl = DataLoader(valid_dataset, batch_size=8, shuffle=True)


        total_iou = 0.0
        total_batch_size = 0
        max_slice_list = []
        total_dice = 0
        total_dice_slice = 0
        counter = 0
        dist = 0
        #CHANGE
        for co,(input_data,text,labels,masks) in enumerate(test_dl):
        # for co,(input_data,_,labels,masks,_) in enumerate(test_dl):
            # counter+=1
            print("shapees",labels.shape,masks.shape,input_data.shape)

            del labels
            
            
            input_data=input_data.float().cuda()
            masks=masks.cuda()
            

            

            #CHANGE
            preds,attn = model(input_data.float())#,input_id,mask) #

            print("predsss",preds.shape)
            probs = torch.sigmoid(preds).squeeze(0)
            
            binary_preds = (probs >= 0.5).squeeze().to(torch.long)
            print("biiiiii",binary_preds)
            pred_idx =(probs >= 0.5).long().unsqueeze(0)
            
            
           
            
            attention_weights = F.interpolate(attn.mean(1).view(attn.shape[0],8,8,5).unsqueeze(1), (240,240,155),mode='trilinear',align_corners=True)
            
            print("pred_idx",probs,pred_idx, input_data.shape , masks.shape,probs.shape,pred_idx.shape)
            masks = masks.unsqueeze(1)
            
            

            
           
            attrs = attention_weights
            
            threshold =1e-2
           
            heatmap = normalize_heatmap(attrs)

            print("uniqq", heatmap.unique())
            

            print("maxandmin",heatmap.max(),heatmap.min())
            
            heatmap = (attrs).float()

            batch_size = heatmap.shape[0]
            masks_slice, heatmap_slice,MRI_slice = main_slice(masks,heatmap,input_data)
            
            binarized_heatmap = (heatmap > threshold).float()
            binarized_heatmap_slice = (heatmap_slice > threshold).float()
            ### plotting 
            rotation = Affine2D().rotate_deg(90)
            fig, axes = plt.subplots(1,4, figsize=(19, 7))

        
            mri_image_np = MRI_slice.cpu().numpy()
            heatmap_np = heatmap_slice.cpu().numpy()

            

            # Apply the heatmap as an overlay on the MRI image
            overlay_image = mri_image_np + heatmap_np#[..., None]

            # Plot segmentation mask
            axes[0].imshow(np.rot90(MRI_slice[0].squeeze(0).cpu()), cmap='gray')
            # axes[0].set_title('MRI Slice',fontsize=26)
            axes[0].axis('off')
            
            axes[1].imshow(np.rot90(masks_slice[0].squeeze(0).cpu()), cmap='gray')
            # axes[1].set_title('Segmentation Mask Slice',fontsize=26)
            axes[1].axis('off')

             
            axes[2].imshow(np.rot90(heatmap_slice[0].squeeze(0).cpu()), cmap='viridis')
            
            # axes[2].set_title('Heatmap Slice',fontsize=26)
            axes[2].axis('off')

            
            axes[3].imshow(np.rot90(overlay_image[0].squeeze(0)), cmap='gray')
            axes[3].imshow(np.rot90(heatmap_np[0].squeeze(0)), cmap='jet', alpha=0.5)
            
            axes[3].axis('off')
            # Adjust layout and save the figure
            plt.tight_layout()
            
              
            plt.savefig(f"sth/plots{co}_bs1_saliency_ii.png")
            

            plt.close()
           
            
                
            soft_dice =dice_coefficient(binarized_heatmap, masks)
            dice = dice_coefficient( binarized_heatmap_slice , masks_slice)
            

          
            print("soft_dice",soft_dice)
            print("dice",dice)
       
            del attrs
            batch_size = heatmap.size(0)
            total_batch_size += batch_size
            
            del input_data
            print("batch",batch_size)
            counter+=1
            total_dice+=soft_dice
            total_dice_slice+=dice
            
        print("hiiii", heatmap.shape)   
        
        average_iou = total_dice/counter
        IOU_list.append(average_iou)
        average_iou_slice = total_dice_slice/counter
        
        slice_list.append(average_iou_slice)
        
    
    

    print("IOU_list_volume",IOU_list)
    print("Mean_IOU_list_volume",sum(IOU_list)/len(IOU_list))
    print("IOU_list_slice",slice_list)
    print("Mean_IOU_list_slice",sum(slice_list)/len(slice_list))
    












