import torch
import pandas as pd
from captum.attr import LayerGradCam,GuidedGradCam,Lime
import matplotlib.pyplot as plt
from fused_model_copy import generate_model
from fused_model import downstream_image_classifier,SelfAttention,DepthAttention
from captum.attr import IntegratedGradients,LRP,FeatureAblation,ShapleyValues,FeaturePermutation,Saliency,Occlusion
import torch.nn as nn
import numpy as np
from PIL import Image
import os 
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from fused_dataset import split_dataset_cv,process_excel,BertDataset,Eval_new_sk
import pickle
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from fused_model import image_text,ResNet_attention,BasicBlock, ResNet_attention2
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
# from torchvision.transforms.functional import ssim

# file = open('input_tokenized_text', 'rb') #important
    
# # dump information to that file

# df_text_excel = pickle.load(file)

# # close the file
# file.close()
def show_attr(attr_map):
    attr_map = attr_map[0,:,:,:,70]
    u,a = viz.visualize_image_attr(
        attr_map.cpu().permute(1, 2,0).numpy(),  # adjust shape to height, width, channels 
        method='heat_map',
        sign='all',
        show_colorbar=True)
    u.savefig('foo.png') 
    # viz._repr_html_()

    # # # components.v1.html(raw_html)
    # with open("output_mai.html", "w") as file:
    #     file.write(a.data)

def center_of_mass(image):
    """
    Calculate the center of mass for a single image.
    """
    # Create coordinate grids
    grid_x, grid_y = torch.meshgrid(torch.arange(image.size(1)), torch.arange(image.size(2)))

    # Calculate mass
    mass = image.sum()

    # Calculate center of mass
    com_x = (grid_x.cuda() * image).sum() / mass
    com_y = (grid_y.cuda() * image).sum() / mass

    return com_x, com_y


def batch_center_of_mass(images):
    """
    Calculate the center of mass for a batch of images.
    """
    
    batch_com_x, batch_com_y = [], []
    for image in images:
        # Calculate center of mass for each image
        com_x, com_y = center_of_mass(image)
        batch_com_x.append(com_x)
        batch_com_y.append(com_y)
        
        return batch_com_x,batch_com_y
    

def _relabel(input):
    _, unique_labels = np.unique(input.cpu(), return_inverse=True)
    return unique_labels.reshape(input.shape)

def _iou_matrix(gt, seg):
    # relabel gt and seg for smaller memory footprint of contingency table
    gt = _relabel(gt)
    seg = _relabel(seg)

    # get number of overlapping pixels between GT and SEG
    n_inter = contingency_table(gt, seg).A

    # number of pixels for GT instances
    n_gt = n_inter.sum(axis=1, keepdims=True)
    # number of pixels for SEG instances
    n_seg = n_inter.sum(axis=0, keepdims=True)

    # number of pixels in the union between GT and SEG instances
    n_union = n_gt + n_seg - n_inter

    iou_matrix = n_inter / n_union
    # make sure that the values are within [0,1] range
    assert 0 <= np.min(iou_matrix) <= np.max(iou_matrix) <= 1

    return iou_matrix
def soft_dice_coefficient(y_true, y_pred, epsilon=1e-6):
    # print("y_pred",y_true.shape,y_pred.shape)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    intersection = torch.sum(y_true_flat * y_pred_flat)
    union = torch.sum(y_true_flat) + torch.sum(y_pred_flat)
    dice_coeff = (2. * intersection + epsilon) / (union + epsilon)
    return dice_coeff

def dice_coefficient(y_pred, y_true, smooth=1.0):
    # print("preeeeeeeeeeee",predicted.shape,target.shape)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    intersection = torch.sum(y_pred_flat * y_true_flat)
    union = torch.sum(y_pred_flat) + torch.sum(y_true_flat) + smooth
    dice = (2.0 * intersection + smooth) / union
    return dice

def iou(predicted, target, smooth=1.0):
    # predicted = predicted.view(-1)
    # target = target.view(-1)
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) - intersection + smooth
    iou_score = (intersection + smooth) / union
    return iou_score

def normalize_mask(mask):
    min_val = np.min(mask)
    max_val = np.max(mask)
    normalized_mask = (mask - min_val) / (max_val - min_val)
    return normalized_mask

def calculate_3d_iou(box1, box2):
    # box1 and box2 should be in the format [x1, y1, z1, x2, y2, z2]

    # Calculate the intersection volume
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    z1_intersection = max(box1[2], box2[2])
    x2_intersection = min(box1[3], box2[3])
    y2_intersection = min(box1[4], box2[4])
    z2_intersection = min(box1[5], box2[5])

    intersection_volume = max(0, x2_intersection - x1_intersection) * max(0, y2_intersection - y1_intersection) * max(0, z2_intersection - z1_intersection)

    # Calculate the volume of each bounding box
    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])

    # Calculate the Union volume
    union_volume = box1_volume + box2_volume - intersection_volume

    # Calculate IoU
    iou = intersection_volume / union_volume

    return iou


# data=process_excel(df_text_excel)
# data.index=range(data.shape[0])
# image_folder="/hpf/largeprojects/fkhalvati/Projects/SickKids_Brain_Preprocessing/preprocessed_all_seq_kk_july_2022"
# dataset = BertDataset(data,image_folder)
# test_dl = DataLoader(dataset, batch_size=4)

inplanes=[64, 128, 256, 512]
# model = image_text()
# model=generate_model(model_depth=18, n_classes=1039,inplanes=inplanes)
# weight_path = "/hpf/largeprojects/fkhalvati/Sara/sk_results"#/image_text_gen_fold__02"#pLGG_results/image_text/whole_MRI_classification__valid_fold__0"#"#.pth"#"/hpf/largeprojects/fkhalvati/Sara/sk_results/lgg_gen_fold__0.pth"
weight_path = "/hpf/largeprojects/fkhalvati/Sara/pLGG_results/image_text"
weight_path1 = "/hpf/largeprojects/fkhalvati/Sara/pLGG_results"
# model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
# model.fc = nn.Linear(512, 1)



# # 
#####CHANGE
model=downstream_image_classifier()
model.cnn.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
model.cnn.fc=nn.Linear(512, 256)
model.cnn.fc1=nn.Linear(256,1)
model.cnn.self_attention=SelfAttention(512)
# model.cnn.self_attention2=SelfAttention()
# model.cnn.depth_attention=DepthAttention(512)

# model = medcam.inject(model.cuda(), output_dir="/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/", save_maps=True)
# os.umask(0) 
# os.chmod('/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/cnn.depth_attention', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


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
# n_interpret_features = 2
# def iter_combinations(*args, **kwargs):
#     for i in range(2 ** n_interpret_features):
#         # print("eyyyy",torch.tensor([int(d) for d in bin(i)[2:].zfill(n_interpret_features)]))
#         yield torch.tensor([int(d) for d in bin(i)[2:].zfill(n_interpret_features)]).unsqueeze(0).cuda()
# exp_eucl_distance = get_exp_kernel_similarity_function('cosine', kernel_width=1000)
# lr_lime = Lime(
#     model, 
#     interpretable_model=SkLearnLasso(alpha=0.1),#SkLearnLinearRegression(),#SkLearnLinearRegression(),#SkLearnLasso(alpha=0.08),  # build-in wrapped sklearn Linear Regression
#     similarity_func=exp_eucl_distance,
#     perturb_func=iter_combinations
# )
##############3pretrain


split_path = "/hpf/largeprojects/fkhalvati/Sara/lgg/splits/new_split"
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

# def calculate_iou(mask1, mask2):
#     mask1,mask2=mask1.cpu(),mask2.cpu()
#     intersection = torch.logical_and(mask1, mask2).sum()
#     union = torch.logical_or(mask1, mask2).sum()
#     print("union",union)
#     iou = intersection.float() / union.float()
#     return iou
# # #######3captum

# model=model.cuda()
# model.eval()
# # for module in model.modules():
# #     if isinstance(module, torch.nn.ReLU):
# #         module.inplace=False
# # with torch.set_grad_enabled(False):

    
# #     test_true = []
# #     test_estimated = []
    
# #     for images,text,labells,masks in test_dl:
        

# #         # images, labells = images.cuda(), labells.cuda()
        
# #         # masks=masks.cuda()
# #         pred = model(images.float())
    
# #         prob = torch.sigmoid(pred)
# #         for i in range(len(labells.tolist())):
# #             test_true.append(labells.tolist()[i])#[0])
# #             test_estimated.append(prob.tolist()[i])#[0])
#         # if epoch==args.num_epochs-1:
#         #     print("test AUC:", model_eval(model,test_dl))

# # test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
# # print("test_auc",test_auc)
# layer_gradcam = LayerGradCam(model, model.layer4[-1].conv2)
# #################33 5-fold cv
# lime = Lime(model)
# # total_iou = 0.0
# # total_batch_size = 0
# # for images,text,labels,masks in valid_dl:
# #     # model.layer1=model.layer1.cuda()
# #     # model.layer2=model.layer2.cuda()
# #     # model.layer3=model.layer3.cuda()
# #     del labels
# #     masks=masks.unsqueeze(1)
# #     print("shape",masks.shape)
# #     # print("mask",torch.unique(masks))
# #     # input_data=images.clone().detach().requires_grad_(True)
# #     images=images.cuda()
    
# #     input_data=images.requires_grad_(True)#.clone()
# #     mask = text['attention_mask']
# #     input_id = text['input_ids'].squeeze(1)

# #     input_id=input_id.cuda()
# #     mask=mask.cuda()
    
# #     preds = model(images.float(),input_id,mask)
# #     probs = torch.sigmoid(preds)
    
# #     # print("probs",probs.shape)
# #     # print("labels",labels.shape)
    
        
# #     # binary_preds = (probs >= 0.5).squeeze().to(torch.long)#.int()
# #     # print("binary_pred",binary_preds.shape)
# #     # attribution = layer_gradcam.attribute(input_data.float())#,target=binary_preds)
#         attribution = lime.attribute(input, target=probs, n_samples=200)
# #     attribution = layer_gradcam.attribute(input_data.float(),target=probs.squeeze().to(torch.long),additional_forward_args=(input_id,mask))
    
# #     # print("attribution",attribution.shape)
# #     # plt.imshow(attribution.squeeze()[2,:,:,2].detach().numpy())
# #     # plt.savefig(os.path.join('/hpf/largeprojects/fkhalvati/Sara/sk_results',f"heatmapimage__{0}.PNG"))
# #     # upsampled_attr = LayerAttribution.interpolate(attribution, (240,240,155))
# #     upsampled_attr = F.interpolate(attribution, (240,240,155),mode='trilinear')#,align_corners=True)
# #     # print("attribution",attribution.shape)
# #     # masks = F.interpolate(masks, (15,8,5),mode='trilinear')
# #     # print("uniq",torch.unique(masks))
    
# #     # print("upsampled_attr",upsampled_attr.shape)
# #     heatmap = upsampled_attr.detach()#torch.abs(upsampled_attr)#.squeeze()#.detach().numpy()
# #     del images,attribution
# #     # print("heatmap shape",heatmap.shape)
# #     # print("uniq2",torch.unique(heatmap))
# #     # print("1",heatmap)
# #     # print("2",torch.abs(heatmap))
# #     # print("maxx",torch.max(heatmap,dim=0))
# #     # print("maxx",torch.max(heatmap,dim=1))
# #     # print("maxx",torch.max(heatmap,dim=2))
# #     # print("maxx",torch.max(heatmap,dim=3))
# #     # prin/torch.max(heatmap,dim=4))
# #     # print("heatmap",heatmap.shape)
# #     # heatmap=heatmap.astype(np.uint8)
# #     # for i in range(155):
        
# #     #     plt.imshow(heatmap[2,:,:,i])
# #     #     plt.savefig(os.path.join('/hpf/largeprojects/fkhalvati/Sara/sk_results',f"heatmapimage{i}.PNG"))
# # #     # break
# #     batch_size = heatmap.size(0)
# #     total_batch_size += batch_size
# #     threshold=100
# #     for i in range(batch_size):
# #         # print("batch_size",batch_size)
# #         heatmap_mask = heatmap[i]  # Heatmap for a single MRI image
# #         masks=masks.cuda()
# #         segmentation_mask = masks[i]  # Manual segmentation for the same MRI image
# #         segmentation_mask_reversed = (segmentation_mask <= 0).float()
# #         # heatmap_binary = (heatmap_mask > threshold).float()  # Apply threshold to convert to binary mask
# #         # segmentation_binary = (segmentation_mask > 0).float()  # Assuming segmentation values > 0 indicate foreground
# #         # segmentation_binary = segmentation_mask .float()
# #         # iou = calculate_iou(heatmap_binary, segmentation_binary)
# #         intersection = heatmap_mask * segmentation_mask
# #         un = heatmap_mask * segmentation_mask_reversed
# #         # print("shapeeee",intersection.shape)
# #         iou = torch.sum(intersection) / torch.sum(un)
# #         total_iou += iou
# # # print("total_batch_size",total_batch_size)
# # average_iou = total_iou / total_batch_size

# # print(f"Average IoU: {average_iou.item()}")
# 5-fold cv

# lime
# v_list=[1,4,2,2,3] #MRI+report
# v_list=[0,1,0,4,1]  #MRI
# v_list=[2,1,2,4,4] # MRI_pretrained
# IOU_list=[]
# for i in [4]:#range(5):
#     # pretrain = torch.load(os.path.join(weight_path,f"image_text_gen_fold__{i}{v_list[i]}" ))#,map_location=torch.device('cpu'))
#     # pretrain = torch.load(os.path.join(weight_path,f"whole_MRI_classification__valid_fold__{i}{v_list[i]}.pth" ))

#     # pretrain = torch.load(os.path.join(weight_path,f"whole_MRI_withoutlayer3_classification__valid_fold__{i}{v_list[i]}.pth" ))
#     pretrain = torch.load(os.path.join(weight_path,f"whole_MRI_withoutlayer3_classification__resnet18_23_valid_fold__{i}{v_list[i]}.pth" ))
#     # pretrain = torch.load(weight_epoch)

#     pretrain_cnn={k: v for k, v in pretrain.items() if k.startswith("cnn") and "local_embed.conv1d_1" not in k and "local_embed.conv1d_2" not in k and "local_embed" not in k and "global_embed" not in k and "self_attention" not in k}#and "fc" not in k}
#     pretrain_cnn = {k.replace('cnn.', ''): v for k, v in pretrain_cnn.items()}

#     model.cnn.load_state_dict(pretrain_cnn)#***
#     # model.load_state_dict(pretrain)#***
    

#     model.cnn.cuda()
#     model.eval()
    
#     layer_gradcam = LayerGradCam(model, model.cnn.layer4[-1].conv2)


#     file = open(os.path.join(split_path,f'train_dataset_MRI({i}, {v_list[i]})'), 'rb') #important
        
#     # dump information to that file

#     train_dataset = pickle.load(file)

#     # close the file
#     file.close()

#     file = open(os.path.join(split_path,f'valid_dataset_MRI({i}, {v_list[i]})'), 'rb') #important
        
#     # dump information to that file

#     valid_dataset = pickle.load(file)

#     # close the file
#     file.close()

#     file = open(os.path.join(split_path,f'test_dataset_MRI{i}'), 'rb') #important
        
#     # dump information to that file

#     test_dataset = pickle.load(file)

#     # close the file
#     file.close()


#     test_dl = DataLoader(test_dataset, batch_size=1)
#     train_dl =DataLoader(train_dataset, batch_size=1, shuffle=True)        
#     valid_dl = DataLoader(valid_dataset, batch_size=1, shuffle=True)


#     total_iou = 0.0
#     total_batch_size = 0
#     for counter,(input_data,text,labels,masks) in enumerate(test_dl):

#         del labels#,text
#         # masks=masks.unsqueeze(1)
        
#         input_data=input_data.cuda()
        
#         input_data=input_data.requires_grad_(True).clone()
        
#         # mask = text['attention_mask']
#         # input_id = text['input_ids'].squeeze(1)

#         # input_id=input_id.cuda()
#         # mask=mask.cuda()
        
#         preds = model(input_data.float())#,input_id,mask)
#         probs = torch.sigmoid(preds)
        
       
        
#         # binary_preds = (probs >= 0.5).squeeze().to(torch.long)#.int()
#         # print("binary_pred",binary_preds.shape)
#         # attribution = layer_gradcam.attribute(input_data.float())#,target=binary_preds)
#         attribution = lime.attribute(input_data.float(),target=probs.squeeze().to(torch.long))#,additional_forward_args=(input_id,mask))

#         # Normalize the attribution for visualization

#         ######################################################################################################################3
#         # attribution = torch.relu(attribution)  # Apply ReLU
#         # attribution /= torch.max(attribution)  # Scale between 0 and 1
#         ##########################################################################################################################
        
#         # print("attribution",attribution.shape)
#         # plt.imshow(attribution.squeeze()[2,:,:,2].detach().numpy())
#         # plt.savefig(os.path.join('/hpf/largeprojects/fkhalvati/Sara/sk_results',f"heatmapimage__{0}.PNG"))
#         # upsampled_attr = LayerAttribution.interpolate(attribution, (240,240,155))
#         upsampled_attr = F.interpolate(attribution, (240,240,155),mode='trilinear')#,align_corners=True)
#         # print("attribution",attribution.shape)
#         # masks = F.interpolate(masks, (15,8,5),mode='trilinear')
#         # print("uniq",torch.unique(masks))
        
#         # print("upsampled_attr",upsampled_attr.shape)
#         heatmap = upsampled_attr.detach()#torch.abs(upsampled_attr)#.squeeze()#.detach().numpy()
#         # del images,attribution
#         # print("heatmap shape",heatmap.shape)
#         # print("uniq2",torch.unique(heatmap))
#         # print("1",heatmap)
#         # print("2",torch.abs(heatmap))
#         # print("maxx",torch.max(heatmap,dim=0))
#         # print("maxx",torch.max(heatmap,dim=1))
#         # print("maxx",torch.max(heatmap,dim=2))
#         # print("maxx",torch.max(heatmap,dim=3))
#         # print("maxx",torch.max(heatmap,dim=4))
#         # print("heatmap",heatmap.shape)
#         # heatmap=heatmap.astype(np.uint8)
#         if i==0 and counter==0:
#             for i in range(155):
#                 # print("h",heatmap.squeeze()[:,:,i].shape)
#                 plt.imshow(heatmap.squeeze()[:,:,i].cpu())
#                 plt.savefig(os.path.join('/hpf/largeprojects/fkhalvati/Sara/sk_results',f"heatmapimage{i}.PNG"))
#     #     # break
#         batch_size = heatmap.size(0)
#         total_batch_size += batch_size
#         threshold=100
#         del input_data
#         for i in range(batch_size):
#             # print("batch_size",batch_size)
#             heatmap_mask = heatmap[i]  # Heatmap for a single MRI image
#             masks=masks.cuda()
#             segmentation_mask = masks[i]  # Manual segmentation for the same MRI image
#             segmentation_mask_reversed = (segmentation_mask <= 0).float()
#             # heatmap_binary = (heatmap_mask > threshold).float()  # Apply threshold to convert to binary mask
#             # segmentation_binary = (segmentation_mask > 0).float()  # Assuming segmentation values > 0 indicate foreground
#             # segmentation_binary = segmentation_mask .float()
#             # iou = calculate_iou(heatmap_binary, segmentation_binary)
#             intersection = heatmap_mask * segmentation_mask
#             un = heatmap_mask * segmentation_mask_reversed
#             # print("shapeeee",intersection.shape)
#             iou = torch.sum(intersection) / torch.sum(un)
#             total_iou += iou
            
#             del segmentation_mask
#             del segmentation_mask_reversed
#             del intersection,un,iou
#         del masks
#         del heatmap
        
#         print("done")
#         # del heatmap_mask,masks,segmentation_mask,segmentation_mask_reversed,intersection,un,iou,preds,probs
# # print("total_batch_size",total_batch_size)
#     average_iou = total_iou / total_batch_size
#     IOU_list.append(average_iou.item())
# print("IOU_list",IOU_list)
# print(f"Average IoU: {np.mean(IOU_list)}")



# heatmaps = []

# Iterate over each slice of the input data
# for images,text,labels in test_dl:
#     input_data=images.clone().detach().requires_grad_(True)
#     for slice_idx in range(155):
#         # Extract the 2D slice from the volume
#         slice_data = input_data[:,:,:,:,slice_idx]

#         # Convert the slice to a tensor and enable gradients
#         tensor_slice = slice_data.unsqueeze(-1).float()
#         # tensor_slice.requires_grad = True

#         # Generate the GradCAM heatmap for the 2D slice
#         heatmap = layer_gradcam.attribute(tensor_slice)
#         print("hhh",heatmap.shape)
#         # Resize the heatmap to match the original slice dimensions
#         # resize = Resize((240,240))#((slice_data.shape[0], slice_data.shape[1]))
#         # resized_heatmap = resize(heatmap)
#         resized_heatmap = LayerAttribution.interpolate(heatmap, (240,240,1))
#         # Convert the resized heatmap to a NumPy array
#         heatmap_array = resized_heatmap.squeeze().detach().numpy()
#         print("after",heatmap_array.shape)
#         # Append the heatmap to the list
#         im=Image.fromarray(heatmap_array[0][:,:])
#         print(heatmap_array[0][:,:].shape)
#         im = im.convert('RGB')
#         im.save(os.path.join('/hpf/largeprojects/fkhalvati/Sara/sk_results',f"heatmapimage{slice_idx}.PNG"))
#         # print("unique",np.unique(heatmap_array[0][:,:]))
#         heatmaps.append(heatmap_array)
    
#     # Stack the heatmaps along the z-axis to form the final 3D heatmap
#     heatmap_3d = np.stack(heatmaps)
#     break



######3 captum


# #### pytorch-gradcam
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image



# for images,text,labels in test_dl:
#     images=images.cuda()
#     images=images.float()
#     labels=labels.cuda()
#     labels=labels.float()
#     input_tensor=images.clone().detach()#.requires_grad_(True)
#     print("input",input_tensor.shape)
#     target_layers= [model.layer4[-1].conv2]
#     with GradCAM(model=model.cuda(), target_layers=target_layers) as cam:
#         grayscale_cam = cam(input_tensor=input_tensor)

  
#     print("camm", grayscale_cam.shape)
#     # print(type(grayscale_cam))
#     # print(grayscale_cam[0, :])
#     print("hhhhhhhhhhh",grayscale_cam[0].shape)

#     # plt.plot()
#     # grayscale_cam=torch.tensor(grayscale_cam)
#     # grayscale_cam=LayerAttribution.interpolate(grayscale_cam, (240,240,1)).detach().numpy()
#     # im=Image.fromarray()#[0])
#     plt.imshow(grayscale_cam[2,:,:,2])
#     plt.savefig(os.path.join('/hpf/largeprojects/fkhalvati/Sara/sk_results',f"heatmapimage{0}.PNG"))
#     # im = im.convert('RGB')
#     # im.save(os.path.join('/hpf/largeprojects/fkhalvati/Sara/sk_results',f"heatmapimage{0}.PNG"))
#     # grayscale_cam = grayscale_cam[0, :]
#     # visualization = show_cam_on_image(input_tensor, grayscale_cam, use_rgb=True)
#     break


### pytorch-gradcam



###med3cam
# from medcam import medcam
# # Inject model with M3d-CAM
# model = medcam.inject(model, output_dir='/hpf/largeprojects/fkhalvati/Sara/sk_results', save_maps=True)
# model.eval()
# for images,text,labels in test_dl:
#     images=images.float()
#     # Every time forward is called, attention maps will be generated and saved in the directory "attention_maps"
#     output = model(images)
    

#####med3cam


# manually

# gradient = None
# feature_maps = None


# def hook_fn(module, grad_in, grad_out):
#     global gradient
#     gradient = grad_out#[0]

# def forward_hook_fn(module, input, output):
#     global feature_maps
#     feature_maps = output

# target_layer = model.layer4[-1].conv2  # Choose the target layer
# target_layer.register_backward_hook(hook_fn)
# target_layer.register_forward_hook(forward_hook_fn)

# model=model.cuda()




# for images,text,labels in test_dl:
#     images=images.cuda()
#     images=images.float()
#     labels=labels.cuda()
#     labels=labels.float()
#     # input_data = images
#     # output = model(input_data)
#     # target_class = torch.argmax(output)
#     # model.zero_grad()
#     # print("target",target_class)
#     # output[0, 0].backward()



#     # weights = torch.mean(gradient, dim=(2, 3, 4))  # Calculate the weights
#     # print("feature_map",feature_maps.shape,weights.shape)
#     # gradcam = torch.zeros_like(feature_maps) * weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Create GradCAM heatmaps
#     # gradcam = torch.sum(gradcam, dim=1, keepdim=True)  # Sum across the channels
#     # gradcam = torch.relu(gradcam)  # Apply ReLU to focus on positive contributions

#     # gradcam_normalized = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())  # Normalize the heatmaps
#     # gradcam_np = gradcam_normalized.detach().cpu().numpy().squeeze()  # Convert to numpy array

#     # plt.imshow(gradcam_np[0,:,:,0], cmap='jet')
#     # plt.axis('off')
#     # plt.savefig(os.path.join('/hpf/largeprojects/fkhalvati/Sara/sk_results',f"heatmapimage{0}.PNG"))
#     # break
#     target_layer = model.layer4[-1].conv2
#     target_layer.register_backward_hook(hook_fn)
#     target_layer.register_forward_hook(forward_hook_fn)
#     # Load and preprocess the input MR image
#     input_image = images  # Load and preprocess your MR image

#     # Forward pass to obtain the output feature map
#     output = model(input_image)
#     print("output",output.shape)
#     # Compute gradients of the target class with respect to the output feature map
#     output[0, 0].backward()

#     # Get the gradients of the target layer
#     gradients = target_layer.grad

#     # Perform global average pooling on the gradients
#     pooled_gradients = torch.mean(gradients, dim=[2, 3, 4])

#     # Get the output feature map of the target layer
#     target_layer_output = output[:, target_class]

#     # Multiply the importance weights with the feature map channels
#     weighted_feature_maps = target_layer_output * pooled_gradients[:, :, None, None, None]

#     # Sum the weighted feature maps along the channel dimension
#     heatmap = torch.sum(weighted_feature_maps, dim=1)

#     # Normalize the heatmap
#     heatmap = torch.relu(heatmap)
#     heatmap /= torch.max(heatmap)
#     break


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
v_list=[2,1,2,4,4] # MRI_pretrained
# v_list=[4,1,0,4,4]
v_list=[4,0,0,0,0]
# v_list = [4,2,2,4,2] # MRI baseline
# v_list = [2,3,2,2,0] # MRI+report baseline
v_list = [0,0,0,0,0]#0,0,0,0] 
IOU_list=[]
slice_list=[]
distance_list = []
with torch.no_grad():
    for i in range(5):
        if i!=0:
            continue
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

        # lime = Lime(model)
        
        
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
        # test_image_folder = os.path.join("/hpf/largeprojects/fkhalvati/Datasets/MedicalImages/BrainData/SickKids/preprocessed_pLGG_EN_Nov2023_KK")
        # df = pd.read_csv("/hpf/largeprojects/fkhalvati/Sara/pLGG_4cohorts_532subs.csv")#,engine='openpyxl')
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

            del labels#,text
            # masks=masks.unsqueeze(1)
            
            input_data=input_data.float().cuda()
            masks=masks.cuda()
            

            
            # input_data=input_data.requires_grad_(True).clone()

            #report_concat
            # mask = text['attention_mask']
            # input_id = text['input_ids'].squeeze(1)

            # input_id=input_id.cuda()
            # mask=mask.cuda()
#report_concat######


            # selected_slices[i, 0, :, :] = input_data[i, 0, :, :, largest_cross_section_indices[i]]

    ## soft dice
            # print("sliceeeeeee",max_slice,len(max_slice))
            # max_slice_list.append(max_slice)
            
            # preds , attn= model(input_data) #
            #CHANGE
            preds,attn = model(input_data.float())#,input_id,mask) #

            # preds = model(input_data.float())
            # preds_mask,attention_weights_mask = model(masks.float())
            print("predsss",preds.shape)
            probs = torch.sigmoid(preds).squeeze(0)#F.softmax(preds, dim=1).squeeze(0)#torch.sigmoid(preds).squeeze(0)
            # probs = F.softmax(preds, dim=1).squeeze(0)
            layer_index = 0 
            # heatmap = attention_weights[layer_index].squeeze().cpu().detach().numpy()
            # masks=masks.unsqueeze(0)

# this part removed
            # attention_weights = attention_weights#.unsqueeze(0).unsqueeze(0)#.squeeze().cpu().detach()#.numpy()
            # attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
# to here

            # attention_weights=attention_weights.reshape(8,8,5)
            # attention_sample = torch.matmul(attention_weights, torch.ones(1,1,320).permute(0, 2, 1).cuda())

            binary_preds = (probs >= 0.5).squeeze().to(torch.long)
            print("biiiiii",binary_preds)
            # layer_gradcam =  LayerGradCam(model, model.cnn.layer4[-1].conv2)#Saliency(model)#LayerGradCam(model, model.cnn.layer4[-1].conv2)#Saliency(model)#FeatureAblation(model)#LayerGradCam(model, model.cnn.layer4[-1].conv2)
            pred_idx =(probs >= 0.5).long().unsqueeze(0)#.squeeze(1)#.to(torch.long)
            # masks[masks == 0.] = 0
            # masks[masks == 1.] = 1
            
            # masks = masks.unsqueeze(1)[0].unsqueeze(0)
            
            # attention_weights = layer_gradcam.attribute(input_data)#,additional_forward_args=(input_id,mask))#,feature_mask=masks.long())#,target=binary_preds)




            # attr = lime.attribute(input_data.float(), target=1, n_samples=100,return_input_shape = True)
            # attention_weights=attention_sample.view(1, 1, 8, 8, 5)
            # print("attt_weights",attention_weights.shape)
            # heatmap = F.interpolate(attention_weights.view(1,1,320,320,1), (240,240,155),mode='trilinear')#,align_corners=True)
            # attention_weights = F.interpolate(attn.view(4,1,1,320,320), (240,240,155),mode='trilinear')#,align_corners=True)
            # attention_weights = F.interpolate(attn.mean(1).view(attn.shape[0],1,8,8,5), (240,240,155),mode='trilinear',align_corners=True)
            # m = nn.Upsample(size=[240,240,155], mode='trilinear')
            # attention_weights = m(attn.mean(1).view(attn.shape[0],1,8,8,5))
            
            attention_weights = F.interpolate(attn.mean(1).view(attn.shape[0],8,8,5).unsqueeze(1), (240,240,155),mode='trilinear',align_corners=True)
            # attention_weights = F.interpolate(attn.mean(1).view(attn.shape[0],1,64,64,25), (240,240,155),mode='trilinear',align_corners=True)
            

            # attention_weights = attention_weights.permute(0,1,3,4,2)
            # print("attentionnnnn",attention_weights.mean(1).view(8,8,5).shape) #'nearest'
            ##HEREEEEEEEEEEEEE
            # probs=probs.squeeze().squeeze().item()
            # pred_idx =(probs >= 0.5).squeeze().to(torch.long) #probs.argmax().unsqueeze(0)
            pred_idx =(probs >= 0.5).long().unsqueeze(0)#.squeeze(1)#.to(torch.long)
            # pred_idx = probs.argmax().unsqueeze(0)
            print("pred_idx",probs,pred_idx, input_data.shape , masks.shape,probs.shape,pred_idx.shape)
            print("seconnd",masks.shape)
            masks = masks.unsqueeze(1)
            # attrs= lr_lime.attribute(
            #     input_data.float().cuda(),
            #     # target=0,
            #     # additional_forward_args=(input_id,mask),
            #     feature_mask=masks.long().cuda(),
            #     # n_samples=4,
            #     perturbations_per_eval=4,
            #     show_progress=True
            #     ).squeeze(0)
            
            # print("hhehhehehhehe",attrs.shape,masks.shape)
            def normalize_heatmap(heatmap):
                min_val = torch.min(heatmap)
                max_val = torch.max(heatmap)
                normalized_heatmap = (heatmap - min_val) / (max_val - min_val)
                return normalized_heatmap

            # attention_weights = normalize_heatmap(attention_weights)

# this part removed
            # heatmap = F.interpolate(attention_weights.mean(2).view(input_data.shape[0],1,8,8,5), (240,240,155),mode='trilinear')#,align_corners=True)
            # print("idddd",attention_weights.shape)
            # attrs = F.interpolate(attention_weights, (240,240,155),mode='trilinear')  # .view(4,1,64,64,1)
            # heatmap = F.interpolate(attention_weights.view(input_data.shape[0],1,64,64,1), (240,240,155),mode='trilinear')#,align_corners=True)
            # heatmap = F.interpolate(attention_weights, (240,240,155),mode='trilinear')#,align_corners=True)
            # attention_weights = cv2.normalize(attention_weights.detach().cpu().numpy(),None,0,255,cv2.NORM_MINMAX)
            # attention_weights = torch.Tensor(attention_weights).cuda()

    # to here        

            # heatmap = F.interpolate(attention_weights, (240,240,155),mode='trilinear')#,align_corners=True)

            ##################3ererrerer
            # nifti_img = nib.Nifti1Image(heatmap.detach().cpu().numpy(), affine=np.eye(4))  # Provide an affine matrix if available

    # Save the NIfTI image to a file
            # nib.save(nifti_img, 'output.nii.gz') 
            # heatmap = F.interpolate(attention_weights.view(1,1,320,320,1), (240,240,155),mode='trilinear')#,align_corners=True)

            # downsampled_masks = F.interpolate(masks, (8,8,5),mode='trilinear')

# from here 
            # attention_weights = normalize_mask(attention_weights)
            attrs = attention_weights
            
            threshold =1e-2#0.009#0.4859#-0.04#0.0003  #0.0003#50
            # threshold:
            # baseline MRI: 0.0003 MRI+report: 1e-7
            #attention:   MRI: 0.001  MR+report: 0.009
            heatmap = normalize_heatmap(attrs)

            # heatmap = torch.clamp(heatmap, 0, 1)
            # print("max_slice",max_slice)
            # # print("heattt",heatmap.shape,masks.shape)
            # print("masksssss2323",masks.shape)
            print("uniqq", heatmap.unique())
            # heatmap = heatmap.unsqueeze(1)

            # heatmap = cv2.normalize(heatmap.detach().cpu().numpy(),None,0,1,cv2.NORM_MINMAX)
            # heatmap = torch.Tensor(heatmap).cuda()
            # thre
            # heatmap = sigmoid_func(heatmap) 
            print("maxandmin",heatmap.max(),heatmap.min())
            # heatmap = (attrs > threshold).float()
            heatmap = (attrs).float()

            batch_size = heatmap.shape[0]
            # for i 
            # print("aya",torch.any(heatmap[0, :, :, :,50] != 0))
            masks_slice, heatmap_slice,MRI_slice = main_slice(masks,heatmap,input_data)
            # heatmap_slice = heatmap[:,:,:,:,50]
            binarized_heatmap = (heatmap > threshold).float()
            binarized_heatmap_slice = (heatmap_slice > threshold).float()
            ### plotting 
            rotation = Affine2D().rotate_deg(90)
            fig, axes = plt.subplots(1,4, figsize=(19, 7))

        # #    # Plot segmentation mask
        #     for i in range(155):    
        #         # plt.imshow(attention_weights[0,0,:,:,i].cpu(), cmap='viridis')
        #         plt.imshow(heatmap[0,0,:,:,i].cpu(), cmap='viridis')

        #         plt.title('Original Gradcam Heatmap')
        #         plt.axis('off')

        #     #     # Plot heatmap
        #     #     # axes[1].imshow(attention_weights[0,0,:,:,4].cpu(), cmap='viridis')
        #     #     # # axes[1].invert_yaxis()
        #     #     # axes[1].set_title('Heatmap Slice')
        #     #     # axes[1].axis('off')
        #     #     axes[1].imshow(heatmap[0,0,:,:,i].cpu(), cmap='viridis')
        #     #     # axes[1].invert_yaxis()
        #     #     axes[1].set_title('Heatmap Slice')
        #     #     axes[1].axis('off')
        #         try: 
        #             plt.savefig(f"/hpf/largeprojects/fkhalvati/Sara/MRI - x/plots{co}_Original_gradcam{i}.png")
        #         except: 
        #             os.chmod(f"/hpf/largeprojects/fkhalvati/Sara/MRI - x/plots{co}_Original_gradcam{i}.png", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                # plt.savefig(f"/hpf/largeprojects/fkhalvati/Sara/MRI - x/plots{co}_Original_gradcam{i}.png")

                # # Plot binarized heatmap
                # axes[2].imshow(binarized_heatmap_slice[0].squeeze(0).cpu(), cmap='gray')
                # axes[2].set_title('Binarized Heatmap Slice')
                # axes[2].axis('off')
            mri_image_np = MRI_slice.cpu().numpy()
            heatmap_np = heatmap_slice.cpu().numpy()

            # Normalize the heatmap values between 0 and 1
            # heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())

            # Apply the heatmap as an overlay on the MRI image
            overlay_image = mri_image_np + heatmap_np#[..., None]

            # Plot segmentation mask
            axes[0].imshow(np.rot90(MRI_slice[0].squeeze(0).cpu()), cmap='gray')
            # axes[0].set_title('MRI Slice',fontsize=26)
            axes[0].axis('off')
            # helper = floating_axes.GridHelperCurveLinear(rotation)#, plot_extents)
            # ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
            # axes[0].set_transform(rotation + axes[0].transData)
            # axes[0].relim()
            # axes[0].autoscale_view()

            axes[1].imshow(np.rot90(masks_slice[0].squeeze(0).cpu()), cmap='gray')
            # axes[1].set_title('Segmentation Mask Slice',fontsize=26)
            axes[1].axis('off')

             # Plot heatmap
            # axes[1].imshow(attention_weights[0,0,:,:,4].cpu(), cmap='viridis')
            # # axes[1].invert_yaxis()
            # axes[1].set_title('Heatmap Slice')
            # axes[1].axis('off')
            axes[2].imshow(np.rot90(heatmap_slice[0].squeeze(0).cpu()), cmap='viridis')
            
            # axes[2].set_title('Heatmap Slice',fontsize=26)
            axes[2].axis('off')

            #  # Plot binarized heatmap
            # axes[3].imshow(binarized_heatmap_slice[0].squeeze(0).cpu(), cmap='viridis')
            # axes[3].set_title('Binarized Heatmap',fontsize=20)
            # axes[3].axis('off')

            
            # print("overlay",overlay_image.shape)
            axes[3].imshow(np.rot90(overlay_image[0].squeeze(0)), cmap='gray')
            axes[3].imshow(np.rot90(heatmap_np[0].squeeze(0)), cmap='jet', alpha=0.5)
            # axes[3].view_init(elev=30, azim=30)
            # axes[3].set_title('MRI with Overlaid Heatmap',fontsize=26)
            axes[3].axis('off')
            # Adjust layout and save the figure
            plt.tight_layout()
            os.chmod("/hpf/largeprojects/fkhalvati/Sara/MRI - x", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            os.umask(0)
              #CHANGE          
            # try: 
            #     plt.savefig(f"/hpf/largeprojects/fkhalvati/Sara/MRI - x/plots{co}_bs1_saliency_ii.png")
            # except: 
            #     os.chmod(f"/hpf/largeprojects/fkhalvati/Sara/MRI - x/plots{co}_bs1_saliency_ii.png", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            #     plt.savefig(f"/hpf/largeprojects/fkhalvati/Sara/MRI - x/plots{co}_bs1_saliency_ii.png")

            plt.close()
            # heatmap_slice = heatmap[:, :, :, :,70]#max_slices]
            
            # masks_slice = masks[:, :, :, :, 70]#max_slices]
            
                #heatmap[:,:,:,:,70]
            # # print("masksss",masks.shape)
            # masks = masks.permute(1,0,2,3,4)
            # masks = cv2.normalize(masks.detach().cpu().numpy(),None,0,1,cv2.NORM_MINMAX)
            # masks = torch.Tensor(masks).cuda()
            # masks_slice = masks[:,:,:,:,70]

            # # heatmap_slice = cv2.applyColorMap(heatmap_slice.detach().cpu().numpy(),cv2.COLORMAP_JET) 

            # print("dimmm",heatmap_slice.shape)
            # import stat
            # print("mean",masks.mean())
            # print("heatttertyertgeh",heatmap_slice.shape,heatmap.shape)
            # heatmap_slice_image = Image.fromarray(heatmap_slice.squeeze().squeeze().detach().cpu().numpy().astype(np.uint8), mode='L')
            # masks_slice_image = Image.fromarray(masks_slice.squeeze().squeeze().detach().cpu().numpy(), mode='L')
            # # os.chmod("/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            # # # os.chmod('/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/heatmap_slice_image.png', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            # # # os.chmod('/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/masks_slice_image.png', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            # # # heatmap_slice_image.save('/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/heatmap_slice_image.png')
            # # # masks_slice_image.save('/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/masks_slice_image.png')
            # plt.imshow(heatmap_slice_image, cmap='viridis')#, vmin=0, vmax=1)
            # plt.axis('off')  # Turn off axis labels
            # # os.chmod('/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/mask_slice_image.png', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            # print("co",co)
            # os.chmod(f'/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/heatmap_{co}_slice_image.png', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            # plt.savefig(f'/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/heatmap_{co}_slice_image.png', bbox_inches='tight', pad_inches=0.0)
            # os.chmod(f'/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/heatmap_{co}_slice_image.png', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
# to here
            # print("heaaatttttttt",heatmap.shape)
            # down_mask=F.interpolate(masks, (heatmap.shape[-3:]),mode='trilinear')#,align_corners=True)
            # if counter==0 and i==0:
            import stat
            # show_attr(heatmap)
            
            # for j in range(155):
            #     layer_index = 0 
            #     print("heatmap",heatmap[2,:,:,:,j].shape)
            #     plt.imshow(masks[2,0,:,:,j].cpu().numpy())#, cmap='viridis')
            #     plt.title(f'Attention Weights for Layer {layer_index}')
            #     # plt.colorbar()
            #     plt.show()
            #     os.chmod(os.path.join('/hpf/largeprojects/fkhalvati/Sara/New folder'), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            #     plt.savefig(os.path.join('/hpf/largeprojects/fkhalvati/Sara/New folder',f"heatmapimage_today{j}.PNG"))
            #     os.chmod(os.path.join('/hpf/largeprojects/fkhalvati/Sara/New folder',f"heatmapimage_today{j}.PNG"), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            # for i in range(155):
            #     heatmap_slice = heatmap[:,:,:,:,i]
            #     masks_slice = masks[:,:,:,:,i]
            #     # print("dimmm",heatmap_slice.shape)
                
            #     # print("mean",masks.mean())
            #     print("masks_slice",masks_slice.shape)
            #     # heatmap_slice_image = Image.fromarray(heatmap_slice.squeeze().detach().cpu().numpy())#, mode='L')
            #     masks_slice_image = Image.fromarray(masks_slice[0].squeeze().detach().cpu().numpy())#, mode='L')
            #     os.chmod("/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            #     # os.chmod('/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/heatmap_slice_image.png', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            #     # os.chmod('/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/masks_slice_image.png', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            #     # heatmap_slice_image.save(f'/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/heatmap_slice_image_{i}.png')
            #     # masks_slice_image.save(f'/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/masks_slice_image_{i}.png')
            #     plt.imshow(masks_slice_image, cmap='gray', vmin=0, vmax=1)
            #     plt.axis('off')  # Turn off axis labels
            #     plt.savefig(f'/hpf/largeprojects/fkhalvati/Sara/lgg_heatmaps/masks_slice_image_{counter}_{i}.png', bbox_inches='tight', pad_inches=0.0)
            # soft_dice = soft_dice_coefficient(masks,heatmap)
            # soft_dice = calculate_3d_iou(masks,heatmap)
            # from pytorch3d.ops import box3d_overlap
            soft_dice =dice_coefficient(binarized_heatmap, masks)#(sigmoid_func(heatmap))
            dice = dice_coefficient( binarized_heatmap_slice , masks_slice)
            

            # x_mask,y_mask = batch_center_of_mass(masks_slice)  
            # x_heat,y_heat = batch_center_of_mass(heatmap_slice)     
            
            
            # x_mask_tensor = torch.tensor(x_mask)
            # y_mask_tensor = torch.tensor(y_mask)
            # x_heat_tensor = torch.tensor(x_heat)
            # y_heat_tensor = torch.tensor(y_heat)

            # # Compute squared differences between x coordinates
            # squared_diffs_x = (x_mask_tensor - x_heat_tensor) ** 2

            # Compute squared differences between y coordinates
            # squared_diffs_y = (y_mask_tensor - y_heat_tensor) ** 2

            # # Sum squared differences along the last dimension (assuming each pair of points is along the last dimension)
            # sum_squared_diffs = squared_diffs_x + squared_diffs_y

            # # Compute square root to get Euclidean distance
            # euclidean_dist = torch.sqrt(sum_squared_diffs)


            # print("eucl",euclidean_dist)
            print("soft_dice",soft_dice)
            print("dice",dice)
        #     # break
            del attrs#attention_weights
            batch_size = heatmap.size(0)
            total_batch_size += batch_size
            # threshold=150
            del input_data
            print("batch",batch_size)
            counter+=1
            total_dice+=soft_dice
            total_dice_slice+=dice
            # if not torch.isnan(euclidean_dist).any():
            #     print("yeess")
            #     dist+=euclidean_dist
        print("hiiii", heatmap.shape)   
        # soft_dice = soft_dice_coefficient(heatmap, masks)

        # for i in range(batch_size):
        #     # print("batch_size",batch_size)
        #     heatmap_mask = heatmap[i]  # Heatmap for a single MRI image
        #     print("maskkk",heatmap_mask.shape)
        #     # masks=masks.cuda()
        #     # print("massskkk",masks.shape)
        #     segmentation_mask = masks[i]  # Manual segmentation for the same MRI image
        #     segmentation_mask_reversed = (segmentation_mask <= 0).float()
        #     # heatmap_binary = (heatmap_mask > threshold).float()  # Apply threshold to convert to binary mask
        #     # segmentation_binary = (segmentation_mask > 0).float()  # Assuming segmentation values > 0 indicate foreground
        #     # segmentation_binary = segmentation_mask .float()
        #     # iou = calculate_iou(heatmap_binary, segmentation_binary)
            
        #     intersection = heatmap_mask * segmentation_mask
        #     un = heatmap_mask * segmentation_mask_reversed
        #     # print("shapeeee",intersection.shape)
        #     iou = torch.sum(intersection) / torch.sum(un)
        #     total_iou += iou
            
        #     del segmentation_mask
        #     del segmentation_mask_reversed
        #     del intersection,un,iou
        #     # del masks
        #     # del heatmap
            
        #     print("done")
        #     # del heatmap_mask,masks,segmentation_mask,segmentation_mask_reversed,intersection,un,iou,preds,probs
        # # print("total_batch_size",total_batch_size)
        # average_iou = total_iou / total_batch_size
        # average_iou = soft_dice / total_batch_size

        # IOU_list.append(average_iou)#.item())
        average_iou = total_dice/counter
        IOU_list.append(average_iou)
        average_iou_slice = total_dice_slice/counter
        average_dist = dist/counter
        distance_list.append(average_dist)
        slice_list.append(average_iou_slice)
        # if i==0:
        #     break
    
    

    print("IOU_list_volume",IOU_list)
    print("Mean_IOU_list_volume",sum(IOU_list)/len(IOU_list))#,np.mean(IOU_list),np.std(IOU_list))
    print("IOU_list_slice",slice_list)
    print("Mean_IOU_list_slice",sum(slice_list)/len(slice_list))#,np.mean(slice_list),np.std(slice_list))
    print("distance",average_dist)
    # print(f"Average IoU: {np.mean(IOU_list)}")












#### lime


# from captum.attr import visualization as viz
# from captum.attr import Lime, LimeBase
# from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
# from captum.attr._core.lime import get_exp_kernel_similarity_function


# n_interpret_features = 2
# def iter_combinations(*args, **kwargs):
#     for i in range(2 ** n_interpret_features):
#         yield torch.tensor([int(d) for d in bin(i)[2:].zfill(n_interpret_features)]).unsqueeze(0)
# exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
# lr_lime = Lime(
#     model, 
#     interpretable_model=SkLearnLasso(alpha=0.08),  # build-in wrapped sklearn Linear Regression
#     similarity_func=exp_eucl_distance,
#     perturb_func=iter_combinations
# )
# # label_idx = output_probs.argmax().unsqueeze(0)
# pred_idx = binary_preds
# attrs = lr_lime.attribute(
#     input_data.unsqueeze(0),
#     target=pred_idx,
#     feature_mask=mask.unsqueeze(0),
#     n_samples=40,
#     perturbations_per_eval=16,
#     show_progress=True
# ).squeeze(0)

# print('Attribution range:', attrs.min().item(), 'to', attrs.max().item())
# print("shapeeeee", attrs.shape)
# def show_attr(attr_map):
#     viz.visualize_image_attr(
#         attr_map.permute(1, 2, 3, 0).numpy(),  # adjust shape to height, width, channels 
#         method='heat_map',
#         sign='all',
#         show_colorbar=True
#     )




# ###### 

# ## TCAV

# from captum.concept import TCAV
# from captum.concept import Concept

# from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
# from captum.concept._utils.common import concepts_to_str


# # gen_data = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Nomogram_study_LGG_data_Nov.27.xlsx",engine='openpyxl')
# #         # # df2 = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Stanford_new_data_09_21.xlsx",engine='openpyxl'))
# #         # tumor_location = gen_data["Location_1"][gen_data["code"]==image_id].values



# for i in range(batch_size):
#     selected_slices[i, 0, :, :] = batch_of_images[i, 0, :, :, largest_cross_section_indices[i]]
# selected_slices = batch_of_images[np.arange(batch_size), :, :, :, largest_cross_section_indices]


# def get_tensor_from_filename(filename):
   
#     img = np.load(filename)#(os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_segmentation.npy"))
#     return torch.Tensor(img)



# image_path = 
# def assemble_concept(name, id, concept_path=image_path):
#     # concept_path = os.path.join(concepts_path, name) + "/"
    
#     # Assuming your images are stored in subdirectories inside the concept_path
#     df_loc = df[df["Location_1"]==name]
#     id_list_loc = list(df_loc["image_id"])
#     subdirectories = [d for d in os.listdir(concept_path) if (os.path.isdir(os.path.join(concept_path, d)) and d in id_list_loc)]
    
#     image_paths = []
#     for subdir in subdirectories:
#         subdir_path = os.path.join(concept_path, subdir,"FLAIR")
#         images_in_subdir = [os.path.join(subdir_path,f) for f in os.listdir(subdir_path) if f=="preprocessed_segmentation.npy"]
#         image_paths.extend(images_in_subdir)
    
#     dataset = CustomIterableDataset(get_tensor_from_filename, image_paths)
#     concept_iter = dataset_to_dataloader(dataset)

#     return Concept(id=id, name=name, data_iter=concept_iter)


# # concepts_path = "data/tcav/image/concepts/"

# supra_concept = assemble_concept("1", 0)
# infra_concept = assemble_concept("2", 1)
# trans_concept = assemble_concept("3", 2)


# model.eval()
# layers=['layer4']
# mytcav = TCAV(model=model,
#               layers=layers,
#               layer_attr_method = LayerIntegratedGradients(
#                 model, None, multiply_by_inputs=False))

# experimental_set_rand = [[supra_concept, infra_concept,trans_concept]]
# tcav_scores_w_random = mytcav.interpret(inputs=input_data.cuda(),
#                                         experimental_sets=experimental_set_rand,
#                                         # target=zebra_ind,
#                                         n_steps=5,
#                                        )

# def format_float(f):
#     return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

# def plot_tcav_scores(experimental_sets, tcav_scores):
#     fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

#     barWidth = 1 / (len(experimental_sets[0]) + 1)

#     for idx_es, concepts in enumerate(experimental_sets):

#         concepts = experimental_sets[idx_es]
#         concepts_key = concepts_to_str(concepts)

#         pos = [np.arange(len(layers))]
#         for i in range(1, len(concepts)):
#             pos.append([(x + barWidth) for x in pos[i-1]])
#         _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
#         for i in range(len(concepts)):
#             val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[concepts_key].items()]
#             _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

#         # Add xticks on the middle of the group bars
#         _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
#         _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
#         _ax.set_xticklabels(layers, fontsize=16)

#         # Create legend & Show graphic
#         _ax.legend(fontsize=16)

#     plt.show()

# plot_tcav_scores(experimental_set_rand, tcav_scores_w_random)    