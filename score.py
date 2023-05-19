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

    all_scores, all_cons_scores, i = [], [], 0
    for inp in tqdm(df.question.unique()):
        outs = list(df[df.question==inp]['sampled_outputs'])
        cons_outs = list(df[df.question==inp]['consistent_outputs'])
        cons_outs = [s.split(':')[-1].strip() for s in cons_outs]
        
        score = cons_scorer.score(inp, outs)
        cons_score = cons_scorer.score(inp, cons_outs)
        
        all_scores.extend([score]*len(outs))
        all_cons_scores.extend([cons_score]*len(outs))

        score_df = pd.DataFrame({
            "score-sampled_outputs": all_scores,
            "score-consistent_outputs": all_cons_scores,
        })
        res_df = pd.concat([df, score_df], axis=1)

        if (i+1)%2==0:
            res_df.to_csv(f"scored_{args.pairwise_sim}_{args.scoring_type}-{args.input_file}", index=False)
        i += 1

    res_df.to_csv(f"scored_{args.pairwise_sim}_{args.scoring_type}-{args.input_file}", index=False)