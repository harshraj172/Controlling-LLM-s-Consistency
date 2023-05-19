import random
import argparse
import numpy as np
import pandas as  pd
from tqdm.auto import tqdm

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from templates import *

def paraphrase(inp, method=1):
    pp_prompt = PromptTemplate(
            input_variables=["method", "sentence"],
            template=PP_TEMPLATE,
        )
    llm = OpenAI(openai_api_key="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",)
    inp_pp = llm(prompt=pp_prompt.format(method=str(method), sentence=inp), stop='\n')
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
        for t in np.arange(0, 1, 0.05):
            llm = OpenAI(openai_api_key="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", temperature=t)
            chain = LLMChain(llm=llm, prompt=prompt)
            out = chain.run({"question":inp,})
            outs.append(out.strip())
    elif type_ == "context":
        llm = OpenAI(openai_api_key="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")
        chain = LLMChain(llm=llm, prompt=prompt)
        for r in range(4):
            inp_pp = paraphrase(inp, method=r+1)
            inp_pps.append(inp_pp)
            out = chain.run({"question":inp_pp,})
            outs.append(out.strip())
    return outs, inp_pps

def ans_via_comparison(inp, outs, type_="sampling"):
    PROMPT_TEMPLATE = \
"""
Question: {question}
For the question above there are several options given, choose one among them which seems to be the most correct."""
    for i in range(len(outs)):
        PROMPT_TEMPLATE += f"""\nOption {i+1}: {outs[i]}"""
    PROMPT_TEMPLATE += """\n\nAnswer:"""  
    prompt = PromptTemplate(
        input_variables=["question",],
        template=PROMPT_TEMPLATE,)
    
    outs = []
    if type_ == "sampling":
        for t in np.arange(0, 1, 0.05):
            llm = OpenAI(openai_api_key="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", temperature=t)
            chain = LLMChain(llm=llm, prompt=prompt)
            out = chain.run({"question":inp})
            outs.append(out.strip())
    elif type_ == "context":
        llm = OpenAI(openai_api_key="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",)
        chain = LLMChain(llm=llm, prompt=prompt)
        for r in range(4):
            inp_pp = paraphrase(inp, method=r+1)
            out = chain.run({"question":inp_pp,})
            outs.append(out.strip())
    return outs




if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='parser to run the script')

    # add arguments
    parser.add_argument('--input_file',
                        type=str,
                        help='path to data in .csv')
    parser.add_argument('--variation_type',
                        type=str,
                        choices=["context", "sampling"],
                        default="sampling",)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    # df = df[:100]
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
            res_df.to_csv(f"result_{args.variation_type}-{args.input_file}", index=False)
            
    res_df.to_csv(f"result_{args.variation_type}-{args.input_file}", index=False)
