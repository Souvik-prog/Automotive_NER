# Automotive_NER
Fine tune a large language model (LLM) for performing named entity recognition (NER) task on an automotive dataset. NER task involves processing unstructured text data to extract useful information/entities.

HuggingFace Link of Fine Tuned Model: https://huggingface.co/Souvik2807/Llama-2-7b-Automotive-finetune-NER/tree/main

Dataset Link1: https://drive.google.com/drive/folders/1ohFCqEKD_J55KUbpxvbCarvywWkyeZP5?usp=sharing
Note: Dataset is stored in data/FLAT_RCL.txt,

Dataset Link2: https://www.nhtsa.gov/nhtsa-datasets-and-apis#recalls
Note: Download NHTSA Recall Dataset

This assignment is broken down in three tasks:

1. First task involves analyzing the data and identifying what are some automotive entities that can be extracted from this data. We are interested in entities related to automotive domain. Some examples could be component, failure issue, vehicle model, corrective action etc.

2. Second task is to use an open source LLM and write the prompt to extract the automotive domain entities from given dataset. We have used Llama2-7b and Flan-T5 in our project. The Llama2-7b LLM has been trained using Zero Shot Learning Technique and Flan-T5 was trained using both Few Shot Learning and Zero Shot Learning technique but responses of the model on Few Shot Learning performed better.

3. Final task is to fine tune the selected LLM on a subset of provided dataset. We have chose Llama2-7b for Fine Tuning. The model was fine-tuned against held out dataset obtained by pre-processing the Few Shot Learning model response of Flan-T5.
   
File Description:
  
  Llama2FT.ipynb: Contains the code for Fine Tuning the Llama2 model
  
  test.py: Run test.py with corresponding input in the console to obtain the output returned by the Fine Tuned Model
  
  
  Data_Preprocessing.ipynb: Notebook that contains code to pre-process the held out data used for Fine Tuning Llama2
  
  model_response.txt: Flan-T5 response after Few Shot Learning
