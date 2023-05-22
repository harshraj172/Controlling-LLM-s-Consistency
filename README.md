# LLM Consistency Scoring

***

## Files
- `generate.py` samples output of a given input to produce variations using different temperatures and input paraphrases. It produces outputs using the default/normal sampling method and the ranking trick to be proposed.
- `score.py` scores the outputs generated using the default/normal sampling method and the ranking trick to be proposed.
- `score_utils` utility functions for `score.py`.
- `example-consistency_paraphrasing.ipynb` proof of concept of the idea.

## Experiments
1. To generate outputs across model sizes and types:
    `python generate.py --input_file truthfulQA-final_labelled.csv --model_name --variation_type sampling`
    Run the above command on different "--model_name":
    - `facebook/opt-125m`
    - `facebook/opt-350m`
    - `facebook/opt-1.3b` 
