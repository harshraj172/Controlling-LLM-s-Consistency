import os
import random
import argparse
import pandas as  pd
from tqdm.auto import tqdm

from score_utils import *

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='parser to run the script')

    # add arguments
    parser.add_argument('--input_file',
                        type=str,
                        help='path to data in .csv')
    parser.add_argument('--pairwise_sim',
                        type=str,
                        choices=["entailment", "llm_prompting"],
                        default=None,
                        help="use only when scoring_type=='cluster_entropy'")
    parser.add_argument('--scoring_type',
                        type=str,
                        choices=["cluster_entropy", "thresholding"],
                        default="cluster_entropy",)
    parser.add_argument('--pair_type',
                        type=str,
                        choices=["Question-Answering", "Translation"],
                        default="Question-Answering",)
    args = parser.parse_args()
            
    df = pd.read_csv(args.input_file)
    
    div = 4 if 'context' in args.input_file else 10
    questions = []
    for i in range(0, len(df.question), div):
        questions.append(df.question[i])
        
    if args.pairwise_sim=="llm_prompting":
        pipe = pipeline(model="google/flan-t5-xl", device_map="auto")
        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        llm = None
    cons_scorer = ConsistencyScoring(args.scoring_type, args.pairwise_sim, args.pair_type, llm)

    # output file
    file_name = f"scored_{args.pairwise_sim if args.pairwise_sim else ''}_{args.scoring_type}-{args.input_file}"
    if os.path.exists(file_name):
        score_df_final = pd.read_csv(file_name)
    else:
        score_df_final = pd.DataFrame()

        
    for inp in tqdm(questions):
        outs = list(df[df.question==inp]['sampled_outputs'])
        cons_outs = list(df[df.question==inp]['consistent_outputs'])
        
        # cleaning
        cons_outs = [re.sub(re.compile(r'Option [0-9]+'), '', str(s)).strip() for s in cons_outs]
        cons_outs = [str(s).split(':')[-1].strip() for s in cons_outs]
        
        try:
            score = cons_scorer.score(inp, outs)
            cons_score = cons_scorer.score(inp, cons_outs)
        except ValueError:
            print('raise ValueError')
            continue
        all_scores = [score]*len(outs)
        all_cons_scores = [cons_score]*len(outs)

        score_df = pd.DataFrame({
            'question': list(df[df.question==inp]['question']),
            'question-paraphrase': list(df[df.question==inp]['question-paraphrase']),
            'sampled_outputs': list(df[df.question==inp]['sampled_outputs']),
            'question-cons_paraphrase': list(df[df.question==inp]['question-cons_paraphrase']),
            'consistent_outputs': list(df[df.question==inp]['consistent_outputs']),
            'correct_outputs': list(df[df.question==inp]['correct_outputs']),
            "score-sampled_outputs": all_scores,
            "score-consistent_outputs": all_cons_scores,
        })
        
        score_df_final = pd.concat([score_df_final, score_df], axis=0)
        score_df_final.to_csv(file_name, index=False)

    score_df_final.to_csv(file_name, index=False)