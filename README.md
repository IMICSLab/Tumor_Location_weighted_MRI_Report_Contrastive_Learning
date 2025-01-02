# Tumor Location-weighted MRI-Report Contrastive Learning: A Framework for Improving the Explainability of Pediatric Brain Tumor Diagnosis

This repository contains the code for our paper titled: "Tumor Location-weighted MRI-Report Contrastive Learning: A Framework for Improving the Explainability of Pediatric Brain Tumor Diagnosis".
Our pretraind Contrastive Learning (CL) weights can be accessible at [here](https://drive.google.com/drive/folders/14yYpOlwhg1c2Ly2E1gK_wB_28nulO403?usp=sharing)

The breakdown of the repository is as follows:
- data: dataset classes and preprocessing functions
- losses: losses and metrics
- model: model-related classes and elements
- explainability.py: the code for extracting model attention maps and computing explainability metrics 
- fused_main.py: the main training code for the downstream classification task
- fused_main_global.py: the main training code for the CL framework

