import argparse
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd
import csv
from langchain import PromptTemplate,  LLMChain


def llama2(text):
    model = "Souvik2807/Llama-2-7b-Automotive-finetune-NER"

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})


    template = """
                Given the defects summary related to Automotive industry take out the entities as "Entity" with which type of entity
                it is as "Label" from data and strictly return result in json format containing ["Entity", "Label"].
                ```{text}```
                Answer:
            """

    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    res = llm_chain.invoke(text)
    print(res)


def main(args):
    sentence = args.paragraph
    llama2(sentence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paragraph', required=True)
    main(parser.parse_args())