# DyRRen: A Dynamic Retriever-Reranker-Generator Model for Numerical Reasoning over Tabular and Textual Data

## Dataset
The FinQA dataset can be accessed on <https://github.com/czyssrs/FinQA>.
Download the FinQA dataset and put it under directory ``FinQADataset``.


## Requirements

- pytorch 1.11.0
- transformers 4.17.0
- pandas
- rank_bm25
- sympy

## Retriever

### Train
To train our retriever, edit scripts/run_finqa.sh to set your own pretrained model path and dataset path. Set "RUN_NAME" to the name of the folder you want to save the checkpoints.Then run:

```
cd retriever && ln -s ../FinQADataset
bash scripts/run_finqa.sh
```

You can choose checkpoint according to the performance of the retriever on dev.

### Inference
To run inference, edit scripts/test_finqa.sh to set your selected checkpoint, then run:

```
bash scripts/test_finqa.sh
```

It will create the retrieval results in the "FinQADataset" directory. To train the generator in the next step, we need to get the retrieval results for all the train, dev and test files. Edit scripts/test_finqa.sh to set "pred_mode" as the file you want the retriever to predict. Then run:

```
python utils/process_predictions.py
```
It will process the retrieval results and then create the input file of the genenrator.

## Reranker-Generator
Place the retrieved sentences files given by retriever under directory ``FinQADataset``.

Script ``scripts/DyRRen.sh`` is the running script for main results of DyRRen:
```
bash scripts/DyRRen.sh
```