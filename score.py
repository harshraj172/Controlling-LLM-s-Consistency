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
                        default="entailment",)
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

    cons_scorer = ConsistencyScoring(args.scoring_type, args.pairwise_sim, args.pair_type)

    for inp in tqdm(df.question.unique()):
        outs = list(df[df.question==inp]['sampled_outputs'])
        cons_outs = list(df[df.question==inp]['consistent_outputs'])
        
        # cleaning
        cons_outs = [re.sub(re.compile(r'Option [0-9]+'), '', s).strip() for s in cons_outs]
        cons_outs = [s.split(':')[-1].strip() for s in cons_outs]
        
        score = cons_scorer.score(inp, outs)
        cons_score = cons_scorer.score(inp, cons_outs)
        
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
        if os.path.exists(f"scored_{args.pairwise_sim}_{args.scoring_type}-{args.input_file}"):
            score_df_final = pd.read_csv(f"scored_{args.pairwise_sim}_{args.scoring_type}-{args.input_file}")
            score_df_final = pd.concat([score_df_final, score_df], axis=0)
        else:
            score_df_final = score_df
        
        score_df_final.to_csv(f"scored_{args.pairwise_sim}_{args.scoring_type}-{args.input_file}", index=False)

    score_df_final.to_csv(f"scored_{args.pairwise_sim}_{args.scoring_type}-{args.input_file}", index=False)