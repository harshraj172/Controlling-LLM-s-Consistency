import numpy as np
from scipy.stats import entropy

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from templates import *

class SimEntail():
    """
    microsoft/deberta-v2-xxlarge-mnli uses
    "id2label": {
        "0": "CONTRADICTION",
        "1": "NEUTRAL",
        "2": "ENTAILMENT"
      },
    """
    def __init__(self, tok_path="microsoft/deberta-base-mnli", model_path="microsoft/deberta-base-mnli", max_len=50):
        super(SimEntail, self).__init__()
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.detection_model.to(device)

    def score(self, inp, out1, out2):
        y_1 = f"Question:{inp}\nAnswer:{out1}"
        y_2 = f"Question:{inp}\nAnswer:{out2}"

        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[2].item()

class SimLLM():
    def __init__(self,):
        super(SimLLM, self).__init__()
        llm = OpenAI(openai_api_key="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", )

        # step 1
        prompt_eval_step1 = PromptTemplate(
                input_variables=["context", "question"],
                template=EVAL_STEP1_TEMPLATE,)
        self.chain_step1 = LLMChain(llm=llm, prompt=prompt_eval_step1)        
        # step 2
        prompt_eval_step2 = PromptTemplate(
                input_variables=["question", "answer1", "answer2"],
                template=EVAL_STEP2_TEMPLATE,)
        self.chain_step2 = LLMChain(llm=llm, prompt=prompt_eval_step2)    

    def score(self, inp, out1, out2):
        out1_step1 = self.chain_step1.run({"context":out1, "question":inp})
        out2_step1 = self.chain_step1.run({"context":out2, "question":inp})

        score = self.chain_step2.run({"question":inp.strip(), "answer1":out1_step1.strip(), "answer2":out2_step1.strip()})
        return 1 if score.strip()=='Yes' else 0


def semantic_clustering(inp, outs, pairwise_sim="entailment", threshold=0.5):
    classifier = SimEntail() if pairwise_sim=="entailment" else SimLLM()
    
    C = [[outs[0]]]
    outs = outs[1:]
    for i in range(len(outs)):
        STORED = False
        for j in range(len(C)):
            s_c = C[j][0]
            left_score = classifier.score(inp, s_c, outs[i])
            right_score = classifier.score(inp, outs[i], s_c)
            
            if left_score>threshold and right_score>threshold:
                STORED = True
                C[j].append(outs[i])
        if not STORED: C.append([outs[i]])
    return C


class ConsistencyScoring():
    def __init__(self, scoring_type, pairwise_sim):
        super(ConsistencyScoring, self).__init__()
        self.scoring_type = scoring_type
        self.pairwise_sim = pairwise_sim

    def score(self, inp, outs):
        if self.scoring_type == "cluster_entropy":
            return self.entropy_score(inp, outs, self.pairwise_sim)
        elif self.scoring_type == "thresholding":
            pass

    def entropy_score(self, inp, outs, pairwise_sim="entailment"):
        # TODO
        # Add exact score via entropy estimate through Monte Carlo
        clusters = semantic_clustering(inp, outs, pairwise_sim)

        pk = np.array([len(c) for c in clusters])/sum([len(c) for c in clusters])
        H = entropy(pk, base=2)
        return H