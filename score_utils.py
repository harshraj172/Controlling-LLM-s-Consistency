import re
import numpy as np
from scipy.stats import entropy
import evaluate
import spacy 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

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
    """
    Using langchain models
    """
    def __init__(self, llm):
        super(SimLLM, self).__init__()
        self.llm = llm 
        
        # step 1
        prompt_eval_step1 = PromptTemplate(
                input_variables=["context", "question"],
                template=EVAL_STEP1_TEMPLATE,)
        self.chain_step1 = LLMChain(llm=llm, prompt=prompt_eval_step1)    
        self.chain_step1.verbose = False    
        # step 2
        prompt_eval_step2 = PromptTemplate(
                input_variables=["question", "answer1", "answer2"],
                template=EVAL_STEP2_TEMPLATE,)
        self.chain_step2 = LLMChain(llm=llm, prompt=prompt_eval_step2)    
        self.chain_step2.verbose = False

    def score(self, inp, out1, out2, type):
        out1_step1 = self.chain_step1.run({"context":out1, "question":inp})
        out2_step1 = self.chain_step1.run({"context":out2, "question":inp})

        score = self.chain_step2.run({"question":inp.strip(), "answer1":out1_step1.strip(), "answer2":out2_step1.strip()})
        return 1 if score.strip()=='Yes' else 0

# class SimLLM():
#     """
#     Using GPT-3.5-turbo
#     """
#     def __init__(self,):
#         super(SimLLM, self).__init__()
#         self.llm = ChatOpenAI(openai_api_key="")

#         system_message_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
#         human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
#         self.chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
 
#     def parse(self, text):
#         """Parse the output of an LLM call."""
#         pattern = re.compile(r"""\b\s"Yes"\s\b""")
#         match = re.search(pattern, text)
#         if match:
#             return 1
#         else:
#             return 0

#     def score(self, inp, out1, out2, type):
#         output = self.llm(self.chat_prompt.format_prompt(input1= inp, 
#                 output1= out1, 
#                 input2= inp, 
#                 output2= out2, 
#                 Type= type).to_messages())

#         return self.parse(output.content)

def semantic_clustering(inp, outs, classifier, pairwise_sim="entailment", pair_type="Question-Answering", threshold=0.5,):
    
    C = [[outs[0]]]
    outs = outs[1:]
    for i in range(len(outs)):
        STORED = False
        for j in range(len(C)):
            s_c = C[j][0]
            left_score = classifier.score(inp, s_c, outs[i], pair_type)
            # right_score = classifier.score(inp, outs[i], s_c, pair_type)
            
            # if left_score>threshold and right_score>threshold:
            if left_score>threshold:
                STORED = True
                C[j].append(outs[i])
        if not STORED: C.append([outs[i]])
    return C


class ConsistencyScoring():
    def __init__(self, scoring_type, pairwise_sim, pair_type, llm):
        super(ConsistencyScoring, self).__init__()
        self.scoring_type = scoring_type
        self.pairwise_sim = pairwise_sim
        self.pair_type = pair_type
        self.llm = llm

    def score(self, inp, outs):
        if self.scoring_type == "cluster_entropy":
            return self.entropy_score(inp, outs, self.pairwise_sim, self.pair_type)
        elif self.scoring_type == "thresholding":
            scorer = MixSimThresholdScore()
            return scorer.get_score(outs)

    def entropy_score(self, inp, outs, pairwise_sim="entailment", pair_type="Question-Answering"):
        classifier = SimEntail() if pairwise_sim=="entailment" else SimLLM(self.llm)
        # TODO
        # Add exact score via entropy estimate through Monte Carlo
        clusters = semantic_clustering(inp, outs, classifier, pairwise_sim, pair_type)

        pk = np.array([len(c) for c in clusters])/sum([len(c) for c in clusters])
        H = entropy(pk, base=2)
        return H
    
class MixSimThresholdScore():
    def __init__(self, tok_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector",\
                 model_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector"):
        super(MixSimThresholdScore, self).__init__()
        self.bleu = evaluate.load('bleu')
        self.bertscore = evaluate.load("bertscore")
        self.pp_detector = PP_Detector(tok_path, model_path)
        self.nli = NLI()
        self.NER = spacy.load("en_core_web_sm")

    def lexical_agreement(self, output_i, output_j):
        if not output_i:
            return 0
        if not output_j:
            return 0
        bleu_score = self.bleu.compute(predictions=[output_i], references=[output_j])
        return bleu_score['bleu'] or 0.0

    def bertscore_agreement(self, output_i, output_j):
        bertscore_score = self.bertscore.compute(predictions=[output_i], references=[output_j], lang='en')
        return bertscore_score['f1'][0]

    def pp_agreement(self, output_i, output_j):
        pp_detector_score = self.pp_detector.score_binary(output_i, output_j)
        return pp_detector_score[1]

    def entailment_agreement(self, output_i, output_j):
        return self.nli.entailed(output_i, output_j)

    def contradiction_agreement(self, output_i, output_j):
        return self.nli.contradicted(output_i, output_j)

    def consistency(self, outputs, agreement_fn, threshold, binary=True):
        agreements = 0
        for i, output_i in enumerate(outputs):
            for j, output_j in enumerate(outputs):
                if i == j:
                    continue
                agreement_score = agreement_fn(output_i, output_j)
                if binary and agreement_score >= threshold:
                    agreements += 1
                elif binary == False:
                    agreements += agreement_score
        if (len(outputs) * (len(outputs) - 1)) == 0:
            return 0
        return (1 / (len(outputs) * (len(outputs) - 1))) * agreements

    def ner_match_score(self, text_i, text_j):
        pro_texti = self.NER(text_i)
        pro_textj = self.NER(text_j)
        num_matches = 0
        all_NERs = []
        for word_i in pro_texti.ents:
            for word_j in pro_textj.ents:
                all_NERs.extend([word_i.text, word_j.text])
                if word_i.text == word_j.text:
                    num_matches += 1
                    break # no multiple match
        if len(all_NERs) == 0:
          return 0.0
        return float(num_matches/len(set(all_NERs)))

    def get_score(self, outputs):
        fns = [
            (self.lexical_agreement, 0.5),
             (self.bertscore_agreement, 0.9),
             (self.pp_agreement, 0.8),
            (self.entailment_agreement, 0.5),
            (self.contradiction_agreement, 0.5),
            (self.ner_match_score, 0.8)
        ]
        con_scores = {}
        for fn, threshold in fns:
            print('Getting score for ', fn.__name__+'_binary')
            con_scores[fn.__name__+'_binary'] = self.consistency(outputs, fn, threshold, True)

        for fn, threshold in fns:
            print('Getting score for ', fn.__name__)
            con_scores[fn.__name__] = self.consistency(outputs, fn, threshold, False)
        return con_scores
    
class PP_Detector():
    def __init__(self, tok_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector", model_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector", max_len=30):
        super(PP_Detector, self).__init__()
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.detection_model.to(device)

    def score_binary(self, y_1, y_2):
        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        # Return probabilites and scores for not paraphrase and paraphrase
        return scores.T[0].item(), scores.T[1].item()


class NLI():
    """
    microsoft/deberta-v2-xxlarge-mnli uses
    "id2label": {
        "0": "CONTRADICTION",
        "1": "NEUTRAL",
        "2": "ENTAILMENT"
      },
    """
    def __init__(self, tok_path="microsoft/deberta-base-mnli", model_path="microsoft/deberta-base-mnli", max_len=30):
        super(NLI, self).__init__()
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.detection_model.to(device)

    def entailed(self, y_1, y_2):
        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[2].item()

    def contradicted(self, y_1, y_2):
        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[0].item()
