# Automotive_NER
Fine tune a large language model (LLM) for performing named entity recognition (NER) task on an automotive dataset. NER task involves processing unstructured text data to extract useful information/entities.

HuggingFace Link of Fine Tuned Model: https://huggingface.co/Souvik2807/Llama-2-7b-Automotive-finetune-NER/tree/main

Dataset Link: https://drive.google.com/drive/folders/1ohFCqEKD_J55KUbpxvbCarvywWkyeZP5?usp=sharing
Note: Dataset is stored in data/FLAT_RCL.txt,

File Description:
  Llama2FT.ipynb: Contains the code for Fine Tuning the Llama2 model
  test.py: Run test.py with corresponding input in the console to obtain the output returned by the Fine Tuned Model
  Data_Preprocessing.ipynb: Notebook that contains code to pre-process the held out data used for Fine Tuning Llama2
  model_response.txt: Flan-T5 response after Few Shot Learning
