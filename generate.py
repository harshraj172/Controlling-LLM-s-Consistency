import random
import argparse
import numpy as np
import pandas as  pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from templates import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def paraphrase(inp, method=1):
    pp_prompt = PromptTemplate(
            input_variables=["method", "sentence"],
            template=PP_TEMPLATE,
        )
    llm = ChatOpenAI(openai_api_key="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",model_name="gpt-3.5-turbo")
    messages = [HumanMessage(content=pp_prompt.format(method=str(method), sentence=inp))]
    inp_pp = llm(messages, stop='\n').content
    return inp_pp.strip()

def produce_output_variations(inp, type_="sampling"): 
    PROMPT_TEMPLATE = \
"""
Question: {question}
Answer the above question in a single sentence.
Answer:"""
# """
# Question: {question}
# Answer the above question in the fewest words possible.
# Answer:"""
    prompt = PromptTemplate(
            input_variables=["question"],
            template=PROMPT_TEMPLATE,)
    
    outs, inp_pps = [], []
    if type_ == "sampling":
        for t in np.arange(0.01, 1, 0.1):
            if args.model_name=="text-davinci-003":
                llm = OpenAI(openai_api_key="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", top_p=0.7, temperature=t)
                chain = LLMChain(llm=llm, prompt=prompt)
                out = chain.run({"question":inp,})
            else:
                input_ids = tokenizer(PROMPT_TEMPLATE.replace("{question}", inp), return_tensors="pt").input_ids
                out_tok = model.generate(input_ids.to(device), max_new_tokens=55, 
                                         top_p=0.7, top_k=0, temperature=t,
                                         do_sample=True, no_repeat_ngram_size=2,)
                out = tokenizer.batch_decode(out_tok, skip_special_tokens=True)[0]
                out = out.replace(PROMPT_TEMPLATE.replace("{question}", inp), '')
            outs.append(out.strip())
    elif type_ == "context":
        llm = OpenAI(openai_api_key="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", top_p=0.7)
        chain = LLMChain(llm=llm, prompt=prompt)
        for r in range(4):
            inp_pp = paraphrase(inp, method=r+1)
            inp_pps.append(inp_pp)
            if args.model_name=="text-davinci-003":
                out = chain.run({"question":inp_pp,})
            else:
                input_ids = tokenizer(PROMPT_TEMPLATE.replace("{question}", inp_pp), return_tensors="pt").input_ids
                out_tok = model.generate(input_ids.to(device), max_new_tokens=55, 
                                         top_p=0.7, top_k=0, temperature=0.6,
                                         do_sample=True, no_repeat_ngram_size=2,)
                out = tokenizer.batch_decode(out_tok, skip_special_tokens=True)[0]
                out = out.replace(PROMPT_TEMPLATE.replace("{question}", inp_pp), '')
            outs.append(out.strip())
    return outs, inp_pps

def ans_via_comparison(inp, outs, type_="sampling"):
    PROMPT_TEMPLATE = \
"""
Question: {question}
For the question above there are several options given, choose one among them which seems to be the most correct."""
    PROMPT_TEMPLATE_SUFFIX = ""
    for i in range(len(outs)):
        PROMPT_TEMPLATE_SUFFIX += f"""\nOption {i+1}: {outs[i]}"""
    PROMPT_TEMPLATE_SUFFIX += f"""\nOption {len(outs)+1}: Don't know the correct answer"""
    PROMPT_TEMPLATE_SUFFIX += """\n\nAnswer:"""  
    PROMPT_TEMPLATE_SUFFIX = PROMPT_TEMPLATE_SUFFIX.replace('{', '{{')
    PROMPT_TEMPLATE_SUFFIX = PROMPT_TEMPLATE_SUFFIX.replace('}', '}}')
    PROMPT_TEMPLATE += PROMPT_TEMPLATE_SUFFIX
    prompt = PromptTemplate(
        input_variables=["question",],
        template=PROMPT_TEMPLATE,)
    
    outs = []
    if type_ == "sampling":
        for t in np.arange(0.01, 1, 0.1):
            if args.model_name=="text-davinci-003":
                llm = OpenAI(openai_api_key="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", top_p=0.7, temperature=t)
                chain = LLMChain(llm=llm, prompt=prompt)
                out = chain.run({"question":inp})
            else:
                input_ids = tokenizer(PROMPT_TEMPLATE.replace("{question}", inp), return_tensors="pt").input_ids
                out_tok = model.generate(input_ids.to(device), max_new_tokens=55, 
                                         top_p=0.7, top_k=0, temperature=t,
                                         do_sample=True, no_repeat_ngram_size=2,)
                out = tokenizer.batch_decode(out_tok, skip_special_tokens=True)[0]
                out = out.replace(PROMPT_TEMPLATE.replace("{question}", inp), '')
            outs.append(out.strip())
    elif type_ == "context":
        llm = OpenAI(openai_api_key="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", top_p=0.7)
        chain = LLMChain(llm=llm, prompt=prompt)
        for r in range(4):
            inp_pp = paraphrase(inp, method=r+1)
            if args.model_name=="text-davinci-003":
                out = chain.run({"question":inp_pp,})
            else:
                input_ids = tokenizer(PROMPT_TEMPLATE.replace("{question}", inp_pp), return_tensors="pt").input_ids
                out_tok = model.generate(input_ids.to(device), max_new_tokens=55, 
                                         top_p=0.7, top_k=0, temperature=0.6,
                                         do_sample=True, no_repeat_ngram_size=2,)
                out = tokenizer.batch_decode(out_tok, skip_special_tokens=True)[0]
                out = out.replace(PROMPT_TEMPLATE.replace("{question}", inp_pp), '')
            outs.append(out.strip())
    return outs




if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='parser to run the script')

    # add arguments
    parser.add_argument('--input_file',
                        type=str,
                        help='path to data in .csv')
    parser.add_argument('--model_name',
                        type=str,
                        default="text-davinci-003")
    parser.add_argument('--variation_type',
                        type=str,
                        choices=["context", "sampling"],
                        default="sampling",)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    df = df[657:].reset_index(drop=True)
    if args.model_name!="text-davinci-003":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.to(device)
    
    all_questions, all_outs, all_inp_pps, all_cons_inp_pps, all_consistent_outs, all_correct_outs = [], [], [], [], [], []
    for i in tqdm(range(len(df))):
        inp = df.question[i]
        correct_ans = df.best_answer[i]
        
        outs, inp_pps = produce_output_variations(inp, type_=args.variation_type)
        
        options, cons_inp_pps = produce_output_variations(inp, type_=args.variation_type)
        cons_outs = ans_via_comparison(inp, options, type_=args.variation_type)
        
        all_questions.extend([inp]*len(outs))
        all_outs.extend(outs)
        all_inp_pps.extend([inp_pps if args.variation_type=="context" else ['']*len(outs)][0])
        all_consistent_outs.extend(cons_outs)
        all_cons_inp_pps.extend([cons_inp_pps if args.variation_type=="context" else ['']*len(outs)][0])
        all_correct_outs.extend([correct_ans]*len(outs))

        res_df = pd.DataFrame({
            "question": all_questions,
            "question-paraphrase": all_inp_pps,
            "sampled_outputs": all_outs,
            "question-cons_paraphrase": all_cons_inp_pps,
            "consistent_outputs": all_consistent_outs,
            "correct_outputs": all_correct_outs,
        })
        
        if (i+1)%1==0:
            res_df.to_csv(f"generated_{args.model_name.replace('/', '')}_{args.variation_type}-{args.input_file}", index=False)
            
    res_df.to_csv(f"generated_{args.model_name.replace('/', '')}_{args.variation_type}-{args.input_file}", index=False)
