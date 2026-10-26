# Cross-lingual Functional Vectors for Emotion Detection in Large Language Models

 
## Abstract 
Function vectors (FVs) have recently emerged as a promising mechanism for steering the behavior of large language models (LLMs) by injecting task-specific latent direction representations derived from in-context demonstrations. While prior studies have shown that FVs can recover task behavior in structured in-context learning settings, their effectiveness on semantically complex tasks and their ability to generalize across languages remain underexplored. We investigate the cross-lingual transferability of FVs using multilingual multi-label emotion recognition as a challenging semantic classification benchmark. Specifically, we examine whether FVs extracted from a source language can steer task behavior in another language under both standard clean and perturbed zero-shot settings without providing demonstrations during inference. Across diverse cross-lingual settings, applying FVs substantially improves performance, suggesting that FVs capture language-agnostic, task-relevant signals rather than purely language-specific lexical patterns, and highlighting their potential as a lightweight and transferable mechanism for multilingual task adaptation. We observe that each LLM exhibits a relatively stable optimal range of attention heads for constructing effective FVs, and the pattern remains consistent across languages. In addition, FVs can partially replicate the task-steering effects of standard few-shot in-context learning while avoiding the computational overhead of processing multiple demonstrations, making them effective for large-scale practical applications.

## Data structure 
```
./data/semeval11/7langs/
├── train
│   └── track_a
│   │   ├── chn.csv
│   │   ├── deu.csv
│   │   ├── eng.csv 
│   │   ├── .... (other languages)
│   │   └── ptbr.csv
│   └── track_b
│       └── eng.csv 
│       ├── .... (other languages)
│       └── ptbr.csv
├── test -> ... (similar to train data)
└── valid -> ... (similar to train data)
```
##  Python ENV 
Init python environment 
```cmd
    conda create --prefix=./env_crossling_fv  python=3.9
    conda activate ./env_crossling_fv 
    pip install -r requirements.txt
```

## Run 

>**Note:** The _detail instruction_ will be updated soon (after the EMNLP2026 conf)!! 
```bash 
# Step 1: Extract AIE matrix for ranking top-k FV heads
bash bash_scripts/run_dump_aie.sh

# Step 2: Inject FV to the fewshot experiments across difference language. 
bash bash_scripts/run_fs.sh
```

## Citation 
```bibtex
@inproceedings{xue-etal-2026-crosslingfv,
  title = "{Cross-lingual Functional Vectors for Emotion Detection in Large Language Models}",
  author = "Xue, Jieying and Nguyen, Phuong Minh and Nguyen, Minh Le and Okada, Shogo",
  booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
  month = oct,
  year = "2026",
  publisher = "Association for Computational Linguistics"
}
```
  
## Credits

This project builds upon our previous work: [Function Vectors in Large Language Models](https://functions.baulab.info/). 

## License
MIT