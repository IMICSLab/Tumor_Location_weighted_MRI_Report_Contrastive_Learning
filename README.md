# Tumor Location-weighted MRI-Report Contrastive Learning

This repository contains the code for our paper titled: "Tumor Location-weighted MRI-Report Contrastive Learning: A Framework for Improving the Explainability of Pediatric Brain Tumor Diagnosis".
The breakdown of the repository is as follows:
- data: dataset classes and huseful functions
- losses: loss and metric code
- model: model-related classes and elements
- explainability.py: the code for extracting model attention maps and computing explainability metrics 
- fused_main.py: the main training code for the downstream classification task
- fused_main_global.py: the main training code for the contrastive learning framework

