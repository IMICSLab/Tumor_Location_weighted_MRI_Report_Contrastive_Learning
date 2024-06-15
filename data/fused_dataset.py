import os
import re
import glob
import torch
import random
import numpy as np
import pandas as pd
# import SimpleITK as sitk
from PIL import Image
from torch.utils.data import Dataset
from ast import literal_eval
import pickle
from transformers import AutoModel, AutoTokenizer,LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import glob
import fnmatch
# from transformers import TransformerLanguageModel#GPT2Tokenizer, GPT2Model#,BioGptForCausalLM, BioGptConfig #BioGptModel
# from transformers.TransformerLanguageModel import BioGptTokenizer
# from transformers import GPT2Tokenizer, GPT2Model




def split_dataset_cv(dataset,train_ratio):
    train_size = int(train_ratio * len(dataset))
 #   validation_size = int(validation_ratio * len(dataset))
    test_size = len(dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,test_size])
    return train_dataset,test_dataset
    


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # set np random seed
    torch.manual_seed(seed_value)  # set torch seed
    random.seed(seed_value)  # set python random seed
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # reproducibility
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False





def collate_fn(data):
    """
    Custom collate_fn dynamic padding, sort by sequence length (descending order)
    Sequences are padded to the maximum length of mini-batch sequences.
    """
    # data.sort(key=lambda x: len(x[0]), reverse=True)##this should be used # x[-1] since we want X_hm. If I were return from dataloader x,y,idx,x_hm,y_hm,demvec, sentence_embedding, i'd do x[-4]
    text, y_label= zip(*data)#, sentence_embedding  #, y_hm, demographic_vec, sentence_embedding = zip(*data)
    # print(text)
    # text = torch.stack(text, dim=0)#,batch_first=True)#
    # print(y_label)
    # print(text[0])
    # y_label = torch.Tensor(y_label)#
    # print(type(text),len(text))
    # text = torch.Tensor(text)
    # y_label = torch.stack(y_label, dim=0)#
    #y_hm = torch.stack(y_hm, dim=0)
    #demographic_vec = torch.stack(demographic_vec, dim=0)
    ##sentence_embedding = torch.stack(sentence_embedding, dim=0)## this should be used
    # if isinstance(X_hm[0], torch.Tensor): X_hm = torch.nn.utils.rnn.pad_sequence(X_hm, batch_first=True)
    # text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
    
    return torch.tensor(text), torch.tensor(y_label)#, sentence_embedding
    #return image, y_label, idx, X_hm, y_hm, demograp



class BertDataset(Dataset):
    def __init__(self,df,image_folder,df_loc):#(self, image_dic,excel_file, image_path_name, class_names)#, image_transform=None)
        # super().__init__()
        self.df=df
        self.image_folder=image_folder
        self.df_loc = df_loc

    def __len__(self):
        return len(self.df)

    # def load_tumor_locations(self,image_id):
    #     gen_data = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Nomogram_study_LGG_data_Nov.27.xlsx",engine='openpyxl')
    #     tumor_location = gen_data["Location_1"][gen_data["code"] == image_id].values  # Extract tumor location
    #     return tumor_location

    def __getitem__(self, idx):
    #    image_name = self.csv_file['dicom_id'].iloc[idx]
        
    #    image, y_label = self.get_image(idx)
        
    #    image=self.image_dic[]     
     
    #    label=self.csv_file[self.[idx]]['label']
    #    return image, label#, idx
        # print(163 in self.df.index)
        raw_text=self.df.loc[idx,"clean_report"]
        # print(raw_text)
        ######tokenizer=AutoTokenizer.from_pretrained("microsoft/biogpt")#("ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section")
        # self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # tokenizer=GPT2Tokenizer.from_pretrained("microsoft/biogpt")#('gpt2')#("healx/gpt-2-pubmed-medium")
        ######tokenizer.pad_token = tokenizer.eos_token
        ######tokenizer.padding_side = 'left'
        # model.resize_token_embeddings(len(tokenizer))
        # tokenizer=AutoTokenizer.from_pretrained("allenai/longformer-base-4096")#TransformerLanguageModel.from_pretrained(
        tokenizer = LongformerTokenizerFast.from_pretrained("yikuan8/Clinical-Longformer")#('allenai/longformer-base-4096')#, max_length = 915)
        # "checkpoints/Pre-trained-BioGPT", 
        # "checkpoint.pt", 
        # "data",
        # tokenizer='moses', 
        # bpe='fastbpe', 
        # bpe_codes="data/bpecodes",
        # min_len=100,
        # max_len_b=1024)
# fix model padding token id
        # model.config.pad_token_id = model.config.eos_token_id

        # tokenized=tokenizer.encode(raw_text)#,return_tensors="pt",padding="max_length",max_length=915)#self.max_sequence_len)#915)#,return_token_type_ids=False)#,truncation=True)#,max_length=1024)
        tokenized=tokenizer.encode_plus(raw_text,return_tensors="pt",padding="max_length", max_length = 1159) #949 #917  #1159   # 1047

        image_id=self.df.loc[idx,"image_id"]
        # gen_data = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Nomogram_study_LGG_data_Nov.27.xlsx",engine='openpyxl')
        #################333 loc
        filtered_df = self.df_loc[self.df_loc['code'] == image_id]
        tumor_location = filtered_df['Location_1'].values
        #################333 loc
        # # df2 = pd.read_excel("/hpf/largeprojects/fkhalvati/Sara/lgg/Stanford_new_data_09_21.xlsx",engine='openpyxl'))
        # tumor_location = gen_data["Location_1"][gen_data["code"]==image_id].values


        # print("loc",tumor_location)
        # tumor_location = self.df.loc[idx,"Location_1"]
        # tumor_location = self.load_tumor_locations(image_id)
        flair_image = np.load((os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_FLAIR.npy")))
        mask = np.load(os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_segmentation.npy"))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        # ROI = torch.tensor(np.multiply(flair_image, mask)).float()#.unsqueeze(0)
        gen_label=self.df.loc[idx,"gen_marker"]
        # print("tyep",type(flair_image),type(tokenized),type(gen_label))
        return flair_image,tokenized,gen_label,mask,tumor_location#mask#,tumor_location
#flair_image

class EvalDataset(Dataset):
    def __init__(self,df1,df2,image_folder):#(self, image_dic,excel_file, image_path_name, class_names)#, image_transform=None)
        # super().__init__()
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
    #    image_name = self.csv_file['dicom_id'].iloc[idx]
        
    #    image, y_label = self.get_image(idx)
        
    #    image=self.image_dic[]     
     
    #    label=self.csv_file[self.[idx]]['label']
    #    return image, label#, idx
        # print(163 in self.df.index)
        
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
            # patient_id =  
            data['gen_marker'] = data["MolecularMarker"].apply(lambda x: 0 if x==2 else 1)
            image_id = data.loc[idx,"Unnamed: 0"]

        # glob.glob(f'{patient_id}*')
        
        
        # image_id=self.df.loc[idx,"image_id"]
        
        flair_image = np.load((os.path.join(self.image_folder, str(image_id).replace(" ",""),"FLAIR", "preprocessed_FLAIR.npy")))
        # mask = np.load(os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_segmentation.npy"))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        # ROI = torch.tensor(np.multiply(flair_image, mask)).float()#.unsqueeze(0)
        gen_label=data.loc[idx,"gen_marker"]
        # print("tyep",type(flair_image),type(tokenized),type(gen_label))
        return flair_image,gen_label


class EvalDataset2(Dataset):
    def __init__(self,df1,image_folder):#(self, image_dic,excel_file, image_path_name, class_names)#, image_transform=None)
        # super().__init__()
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
    #    image_name = self.csv_file['dicom_id'].iloc[idx]
        
    #    image, y_label = self.get_image(idx)
        
    #    image=self.image_dic[]     
     
    #    label=self.csv_file[self.[idx]]['label']
    #    return image, label#, idx
        # print(163 in self.df.index)
       
        data = self.df1
        key_code = data.loc[idx, 'code']
        patient_id = f'{int(key_code)}.St_'
        # print("idddd",patient_id)
        
        # glob.glob(f'{patient_id}*')
        data['gen_marker'] = data.apply(lambda x: self.label_extracter(x['BRAF V600E final'], x['BRAF fusion final']), axis=1)
        image_id = fnmatch.filter(os.listdir(self.image_folder), f'{patient_id}*')[0]
        # print("imageeeeeeeeeeeeeeeeeeee",image_id)
        # image_id=self.df.loc[idx,"image_id"]
        flair_image = np.load((os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_FLAIR.npy")))
        # mask = np.load(os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_segmentation.npy"))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        # ROI = torch.tensor(np.multiply(flair_image, mask)).float()#.unsqueeze(0)
        gen_label=data.loc[idx,"gen_marker"]
        # print("tyep",type(flair_image),type(tokenized),type(gen_label))
        return flair_image,gen_label

class ExtDataset(Dataset):
    def __init__(self,df1,image_folder):#(self, image_dic,excel_file, image_path_name, class_names)#, image_transform=None)
        # super().__init__()
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
        
        # glob.glob(f'{patient_id}*')
        data['gen_marker'] = data.apply(lambda x: self.label_extracter(x['BRAF V600E final'], x['BRAF fusion final']), axis=1)
        image_id = fnmatch.filter(os.listdir(self.image_folder), f'{patient_id}*')[0]
        # print("imageeeeeeeeeeeeeeeeeeee",image_id)
        # image_id=self.df.loc[idx,"image_id"]
        flair_image = np.load((os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_FLAIR.npy")))
        # mask = np.load(os.path.join(self.image_folder, str(image_id), "FLAIR", "preprocessed_segmentation.npy"))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        # ROI = torch.tensor(np.multiply(flair_image, mask)).float()#.unsqueeze(0)
        gen_label=data.loc[idx,"gen_marker"]
        # print("tyep",type(flair_image),type(tokenized),type(gen_label))
        return flair_image,gen_label,gen_label,gen_label,gen_label
def process_excel(df_data):
#     def f(mut, fus):
#         if mut == 1:
#             return 1
#         elif fus ==1:
#             return 0
#         else:
#             return "other"
#     df_data['label'] = df_data.apply(lambda x: f(x['BRAF V600E final'], x['BRAF fusion final']), axis=1)
    # print(df_data["gen_marker"].unique())
    print("data1",df_data.shape)
    # nanmask = np.isnan(df_data["gen_marker"]) uncomment for  binart
    # print("valu_count",df_data["gen_marker"].value_counts())
    # data_SK = df_data[~nanmask]  uncomment for  binart
    data_SK=df_data
    data_SK = data_SK.reindex()
    # print("columnss",df_data.columns)
    # print("finallll funall",df_data["gen_marker"].value_counts())
    data_SK['gen_marker']=data_SK["gen_marker"].apply(lambda x: 0 if x=="other" else x)
    excluded_patients = [9.0, 12.0, 23.0, 33.0, 37.0, 58.0, 74.0, 78.0, 85.0, 121.0, 122.0, 130.0, 131.0, 138.0, 140.0, 150.0,
                         171.0, 176.0, 182.0, 204.0, 213.0, 221.0, 224.0, 234.0, 235.0, 243.0, 245.0, 246.0, 255.0,
                         261.0, 264.0, 274.0, 283.0, 288.0, 293.0, 299.0, 306.0, 309.0,
                         311.0, 312.0, 325.0, 327.0, 330.0, 333.0, 334.0, 347.0, 349.0, 351.0, 352.0, 354.0, 356.0, 359.0,
                         364.0, 367.0, 376.0, 377.0, 383.0, 387.0]

    # Remove exluded patients
    # print(df_data.shape,df_data.columns)
    print("dataaaaaaa",data_SK.shape)
    # data_SK=data_SK[data_SK["gen_marker"]!="other"]   uncomment for binary
    data_SK = data_SK[~data_SK["image_id"].isin(excluded_patients)]
    data_SK = data_SK.reindex()

    # Remove data that we don't need for this analysis
    

    
    # data_SK['label'] = data_SK['Pathology Coded'].apply(lambda x: 1 if x==1.0 else (2 if x==2.0 else 3))
    # data_SK = data_SK.drop(columns=["BRAF V600E final", "BRAF fusion final",'WT', 'NF1',
    #                                             'CDKN2A (0=balanced, 1=Del, 2=Undetermined)', 'FGFR 1', 'FGFR 2',
    #                                             'FGFR 4',
    #                                             'Further gen info', 'Notes', 'Pathology Dx_Original', 'Pathology Coded',
    #                                             'Location_1', 'Location_2', 'Location_Original', 'Gender', 'Age Dx'])
    # Drop rows where the outcome is not mutation or fusion
    #nanmask = np.isnan(data_data_new["label"])
    #data_data_new = data_data_new[~nanmask]
    data_SK = data_SK.reindex()
    
    
    
    # patient_codes = [int(x) for x in list(data_SK["code"].values)]

    # training_labels = dict(zip(patient_codes, list(data_SK["label"].values)))
    # data_SK = data_SK.drop(columns=["label"])


    # Organize the radiomic features into a dictionary with patient codes and corresponding patient features
   # data_data_new.set_index("code", inplace=True)
   # radiomic_features = {}
    #for index, row in data_data_new.iterrows():
     #   radiomic_features[index] = row.values
    # print(data_SK.index)
    return data_SK[["gen_marker","clean_report","image_id"]]#data_SK,training_labels   #"tokenized_index_report"


# class Eval_new_sk(Dataset):
#     def __init__(self,df,image_folder):
#         # super().__init__()
#         self.df=df
        
#         self.image_folder=image_folder

#         name_list = os.listdir(self.image_folder)
        
#         data = self.df[self.df["folder_name"].isin(name_list)]
#         data = data [data["folder_name"].notnull()]
#         self.final_data = data[(data["Gen_marker"]==1)| (data["Gen_marker"]==2)]
#         print("finalldata",self.final_data.shape)
#         self.final_name_list = [element for element in name_list if element in list(self.final_data["folder_name"])]
        
#         # patient_id = f'{int(key_code)}.St_'
#         self.final_data['gen_marker'] = self.final_data["Gen_marker"].apply(lambda x: self.label_extracter(x))
#     def __len__(self):
#         return len(self.final_data)

#     def label_extracter(self,mut_fus):
#         if mut_fus == 1:
#             return 0
#         elif mut_fus ==2:
#             return 1
        


#     def __getitem__(self, idx):
    
        
        
#         image_id = self.final_name_list[idx]#fnmatch.filter(os.listdir(self.image_folder), f'{patient_id}*')[0]
#         # print("iddd",image_id)
#         image_ind = self.final_data.index[self.final_data["folder_name"]==str(image_id)]
#         # print("finall",final_data.loc[449,"folder_name"])
#         # print("fina;",final_data[final_data["folder_name"]=="LGG1413"])
#         # if str(image_id) not in [i for i in list(self.final_data["folder_name"])]:
#         #     print("eeee",image_id)
#         # print("indddd",image_ind.values,image_ind.values[0])
#         flair_image = np.load((os.path.join(self.image_folder, image_id,"FLAIR", "preprocessed_FLAIR.npy")))
#         flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
#         flair_image=torch.tensor(flair_image).unsqueeze(0)
#         gen_label=self.final_data.loc[image_ind.values[0],"gen_marker"]
#         return flair_image,gen_label,gen_label,gen_label,gen_label
        




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
        
        # patient_id = f'{int(key_code)}.St_'
        self.final_data['gen_marker'] = self.final_data["Gen_marker"].apply(lambda x: self.label_extracter(x))
    def __len__(self):
        return len(self.final_data)

    def label_extracter(self,mut_fus):
        if mut_fus == 1:
            return 0
        elif mut_fus ==2:
            return 1
        


    def __getitem__(self, idx):
    
        
        
        image_id = self.final_name_list[idx]#fnmatch.filter(os.listdir(self.image_folder), f'{patient_id}*')[0]
        # print("iddd",image_id)
        image_ind = self.final_data.index[self.final_data["folder_name"]==str(image_id)]
        # print("finall",final_data.loc[449,"folder_name"])
        # print("fina;",final_data[final_data["folder_name"]=="LGG1413"])
        # if str(image_id) not in [i for i in list(self.final_data["folder_name"])]:
        #     print("eeee",image_id)
        # print("indddd",image_ind.values,image_ind.values[0])
        flair_image = np.load((os.path.join(self.image_folder, image_id,"FLAIR", "preprocessed_FLAIR.npy")))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        mask = np.load((os.path.join(self.image_folder, image_id,"FLAIR", "preprocessed_segmentation.npy")))
        gen_label=self.final_data.loc[image_ind.values[0],"gen_marker"]
        return flair_image,gen_label,gen_label,mask,gen_label





# class Eval_new_sk(Dataset):
#     def __init__(self,df,image_folder):
#         # super().__init__()
#         self.df=df
        
#         self.image_folder=image_folder

#         name_list = os.listdir(self.image_folder)
        
#         # data = self.df[self.df["folder_name"].isin(name_list)]
#         # data = data [data["folder_name"].notnull()]
#         self.final_data = df[(df["Gen_marker"]==1)| (df["Gen_marker"]==2)]
#         print("finalldata",self.final_data.shape)
#         # self.final_name_list = [element for element in name_list if element in list(self.final_data["folder_name"])]
        
#         # patient_id = f'{int(key_code)}.St_'
#         self.final_data['gen_marker'] = self.final_data["Gen_marker"].apply(lambda x: self.label_extracter(x))
#         self.final_data.index = range(len(self.final_data))
#     def __len__(self):
#         return len(self.final_data)

#     def label_extracter(self,mut_fus):
#         if mut_fus == 1:
#             return 0
#         elif mut_fus ==2:
#             return 1
        


#     def __getitem__(self, idx):
    
        
        
#         # image_id = self.final_name_list[idx]#fnmatch.filter(os.listdir(self.image_folder), f'{patient_id}*')[0]
#         # print("iddd",image_id)
#         image_id = self.final_data.loc[idx,"folder_name"]
#         # print("finall",final_data.loc[449,"folder_name"])
#         # print("fina;",final_data[final_data["folder_name"]=="LGG1413"])
#         # if str(image_id) not in [i for i in list(self.final_data["folder_name"])]:
#         #     print("eeee",image_id)
#         # print("indddd",image_ind.values,image_ind.values[0])
#         flair_image = np.load((os.path.join(self.image_folder, image_id,"FLAIR", "preprocessed_FLAIR.npy")))
#         flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
#         flair_image=torch.tensor(flair_image).unsqueeze(0)
#         mask = np.load((os.path.join(self.image_folder, image_id,"FLAIR", "preprocessed_segmentation.npy")))
#         gen_label=self.final_data.loc[idx,"gen_marker"]
#         return flair_image,gen_label,gen_label,mask,gen_label


class fewshot_support(Dataset):
    def __init__(self,df,image_folder):
        # super().__init__()
        self.df=df
        
        self.image_folder=image_folder

        name_list = os.listdir(self.image_folder)
        # print("len",len(df),len(name_list))
        # data = self.df[self.df["folder_name"].isin(name_list)]
        # self.data = data[data["folder_name"].notnull()]
        
        self.df = self.df[(self.df["Gen_marker"]==1)| (self.df["Gen_marker"]==2)]
        self.df.index = range(len(self.df))
        # print("finalldata",self.final_data.shape)
        # self.final_name_list = [element for element in name_list if element in list(self.final_data["folder_name"])]
        self.final_name_list=[self.df.loc[0,"folder_name"]]
        label_list = [self.df.loc[0,"Gen_marker"]]
        for i in range(len(self.df)):
            # if element in list(self.data["folder_name"]):
                if self.df.loc[i,"Gen_marker"] in label_list:
                    continue
                self.final_name_list.append(self.df.loc[i,"folder_name"])
                label_list.append(self.df.loc[i,"Gen_marker"])
                # print("label",label)
        
        # patient_id = f'{int(key_code)}.St_'
        # self.data['gen_marker'] = self.data["Gen_marker"].apply(lambda x: self.label_extracter(x))
    def __len__(self):
        return len(self.df)

    # def label_extracter(self,mut_fus):
    #     if mut_fus == 1:
    #         return 0
    #     elif mut_fus ==2:
    #         return 1
        


    def __getitem__(self, idx):
    
        
        
        image_id = self.df.loc[idx,"folder_name"]#fnmatch.filter(os.listdir(self.image_folder), f'{patient_id}*')[0]
        # print("iddd",image_id)
        # image_ind = self.data.index[self.data["folder_name"]==str(image_id)]
        # print("finall",final_data.loc[449,"folder_name"])
        # print("fina;",final_data[final_data["folder_name"]=="LGG1413"])
        # if str(image_id) not in [i for i in list(self.final_data["folder_name"])]:
        #     print("eeee",image_id)
        # print("indddd",image_ind.values,image_ind.values[0])
        flair_image = np.load((os.path.join(self.image_folder, image_id,"FLAIR", "preprocessed_FLAIR.npy")))
        flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))#.unsqueeze(0)
        flair_image=torch.tensor(flair_image).unsqueeze(0)
        
        gen_label=self.df.loc[idx,"Gen_marker"]
        return flair_image,gen_label