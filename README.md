# LLM Consistency Scoring

***

## Files
- `generate.py` samples output of a given input to produce variations using different temperatures and input paraphrases. It produces outputs using the default/normal sampling method and the ranking trick to be proposed.
- `score.py` scores the outputs generated using the default/normal sampling method and the ranking trick to be proposed.
- `score_utils` utility functions for `score.py`.
- `example-consistency_paraphrasing.ipynb` proof of concept of the idea.

## Setup
`pip install -r requirements.txt`
## Experiments
1. To generate outputs across model sizes and types:
    
    `python generate.py --input_file truthfulQA-final_labelled.csv --model_name --variation_type sampling`
    
    `python generate.py --input_file truthfulQA-final_labelled.csv --model_name --variation_type context`
    
    Run the both of the above commands on different `--model_name`:
    - `facebook/opt-125m`
    - `facebook/opt-350m`
    - `facebook/opt-1.3b` 
    - `facebook/opt-6.7b` 
    - `facebook/opt-13b`
    - `EleutherAI/pythia-410m`
    - `EleutherAI/pythia-2.8b`
    - `EleutherAI/pythia-6.9b`
    
2. Score the outputs via the consistency scoring pipeline:

    `python score.py --input_file file_path --pairwise_sim llm_prompting --scoring_type cluster_entropy`
    
    Run the above command with all the files generated from **step 1** specifying their paths in the `--input_file` argument.
