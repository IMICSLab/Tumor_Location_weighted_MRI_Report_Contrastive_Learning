import os
import re
import glob
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from ast import literal_eval
import pickle
from transformers import AutoModel, AutoTokenizer,LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import glob
import fnmatch


class BertDataset(Dataset):
    def __init__(self,df,image_folder,df_loc):#(self, image_dic,excel_file, image_path_name, class_names)#, image_transform=None)
        super().__init__()
        self.df=df
        self.image_folder=image_folder
        self.df_loc = df_loc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
    
        raw_text=self.df.loc[idx,"clean_report"]
       
        tokenizer = LongformerTokenizerFast.from_pretrained("yikuan8/Clinical-Longformer")
        
        tokenized=tokenizer.encode_plus(raw_text,return_tensors="pt",padding="max_length", max_length = 1159) #949 #917  #1159   # 1047

        image_id=self.df.loc[idx,"image_id"]
        filtered_df = self.df_loc[self.df_loc['code'] == image_id]
        tumor_location = filtered_df['Location_1'].values
        flair_image = np.load((os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_FLAIR.npy")))
        mask = np.load(os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_segmentation.npy"))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        # ROI = torch.tensor(np.multiply(flair_image, mask)).float()
        gen_label=self.df.loc[idx,"gen_marker"]
        return flair_image,tokenized,gen_label,mask,tumor_location#mask#,tumor_location

class EvalDataset(Dataset):
    def __init__(self,df1,df2,image_folder):
        super().__init__()
        self.df1=df1
        self.df2=df2
        self.image_folder=image_folder


    def __len__(self):
        return len(self.df1)+len(self.df2)

    def label_extracter(self,mut,fus):
        if mut == 1:
            return 1
        elif fus ==1:
            return 0
        else:
            return "other"


    def __getitem__(self, idx):
    
        
        if idx < len(self.df1):
            # print("idx",idx)
            data = self.df1
            key_code = data.loc[idx, 'code']
            patient_id = f'{int(key_code)}.St_'
            data['gen_marker'] = data.apply(lambda x: self.label_extracter(x['BRAF V600E final'], x['BRAF fusion final']), axis=1)
            image_id = fnmatch.filter(os.listdir(self.image_folder), f'{patient_id}*')[0]
        else:
            data = self.df2
            idx -= len(self.df1)
            if data.loc[idx,"Unnamed: 0"] == "P30":
                idx+=1 
            data['gen_marker'] = data["MolecularMarker"].apply(lambda x: 0 if x==2 else 1)
            image_id = data.loc[idx,"Unnamed: 0"]
        
        flair_image = np.load((os.path.join(self.image_folder, str(image_id).replace(" ",""),"FLAIR", "preprocessed_FLAIR.npy")))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        gen_label=data.loc[idx,"gen_marker"]
        return flair_image,gen_label


class EvalDataset2(Dataset):
    def __init__(self,df1,image_folder):#(self, image_dic,excel_file, image_path_name, class_names)#, image_transform=None)
        super().__init__()
        self.df1=df1
        
        self.image_folder=image_folder


    def __len__(self):
        return len(self.df1)#+len(self.df2)

    def label_extracter(self,mut,fus):
        if mut == 1:
            return 1
        elif fus ==1:
            return 0
        else:
            return "other"


    def __getitem__(self, idx):
   
       
        data = self.df1
        key_code = data.loc[idx, 'code']
        patient_id = f'{int(key_code)}.St_'
        # print("idddd",patient_id)
        
        data['gen_marker'] = data.apply(lambda x: self.label_extracter(x['BRAF V600E final'], x['BRAF fusion final']), axis=1)
        image_id = fnmatch.filter(os.listdir(self.image_folder), f'{patient_id}*')[0]
        flair_image = np.load((os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_FLAIR.npy")))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        gen_label=data.loc[idx,"gen_marker"]
        # print("tyep",type(flair_image),type(tokenized),type(gen_label))
        return flair_image,gen_label

class ExtDataset(Dataset):
    def __init__(self,df1,image_folder):
        super().__init__()
        self.df1=df1
        
        self.image_folder=image_folder


    def __len__(self):
        return len(self.df1)#+len(self.df2)

    def process_excel(self,df):
        excluded_patients = [15,21,24,53,58,83,99,101,104,114,115]
        df_m = df[~df["image_id"].isin(excluded_patients)]
        df_m=df_m[df_m["Subgroup"]== 1 | df_m["Subgroup"]== 2]


    def __getitem__(self, idx):
       
        data = self.df1
        key_code = data.loc[idx, 'Pseudo-MRN / Patient Number']
        patient_id = fnmatch.filter(os.listdir(self.image_folder), f'*{patient_id}')[0]
        # print("idddd",patient_id)
        
        data['gen_marker'] = data.apply(lambda x: self.label_extracter(x['BRAF V600E final'], x['BRAF fusion final']), axis=1)
        image_id = fnmatch.filter(os.listdir(self.image_folder), f'{patient_id}*')[0]
       
        flair_image = np.load((os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_FLAIR.npy")))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        gen_label=data.loc[idx,"gen_marker"]
        # print("tyep",type(flair_image),type(tokenized),type(gen_label))
        return flair_image,gen_label,gen_label,gen_label,gen_label
def process_excel(df_data):

    print("data1",df_data.shape)
    
    data_SK=df_data
    data_SK = data_SK.reindex()
    
    data_SK['gen_marker']=data_SK["gen_marker"].apply(lambda x: 0 if x=="other" else x)
    excluded_patients = [9.0, 12.0, 23.0, 33.0, 37.0, 58.0, 74.0, 78.0, 85.0, 121.0, 122.0, 130.0, 131.0, 138.0, 140.0, 150.0,
                         171.0, 176.0, 182.0, 204.0, 213.0, 221.0, 224.0, 234.0, 235.0, 243.0, 245.0, 246.0, 255.0,
                         261.0, 264.0, 274.0, 283.0, 288.0, 293.0, 299.0, 306.0, 309.0,
                         311.0, 312.0, 325.0, 327.0, 330.0, 333.0, 334.0, 347.0, 349.0, 351.0, 352.0, 354.0, 356.0, 359.0,
                         364.0, 367.0, 376.0, 377.0, 383.0, 387.0]

    data_SK = data_SK[~data_SK["image_id"].isin(excluded_patients)]
    data_SK = data_SK.reindex()
    data_SK = data_SK.reindex()
    return data_SK[["gen_marker","clean_report","image_id"]]#data_SK,training_labels   #"tokenized_index_report"




        





class Eval_new_sk(Dataset):
    def __init__(self,df,image_folder):
        # super().__init__()
        self.df=df
        
        self.image_folder=image_folder

        name_list = os.listdir(self.image_folder)
        
        data = self.df[self.df["folder_name"].isin(name_list)]
        data = data [data["folder_name"].notnull()]
        self.final_data = data[(data["Gen_marker"]==1)| (data["Gen_marker"]==2)]
        print("finalldata",self.final_data.shape)
        self.final_name_list = [element for element in name_list if element in list(self.final_data["folder_name"])]
        self.final_data['gen_marker'] = self.final_data["Gen_marker"].apply(lambda x: self.label_extracter(x))
    def __len__(self):
        return len(self.final_data)

    def label_extracter(self,mut_fus):
        if mut_fus == 1:
            return 0
        elif mut_fus ==2:
            return 1
        


    def __getitem__(self, idx):
    
        
        image_id = self.final_name_list[idx]
        # print("iddd",image_id)
        image_ind = self.final_data.index[self.final_data["folder_name"]==str(image_id)]
        flair_image = np.load((os.path.join(self.image_folder, image_id,"FLAIR", "preprocessed_FLAIR.npy")))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        mask = np.load((os.path.join(self.image_folder, image_id,"FLAIR", "preprocessed_segmentation.npy")))
        gen_label=self.final_data.loc[image_ind.values[0],"gen_marker"]
        return flair_image,gen_label,gen_label,mask,gen_label








