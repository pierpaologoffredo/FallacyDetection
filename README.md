# Argument-based Detection and Classification of Fallacies in Political Debates

In this repository you find the code which was used for the experiments in our paper [Argument-based Detection and Classification of Fallacies in Political Debates](https://aclanthology.org/2023.emnlp-main.684/) (EMNLP 2023).

## Abstract

This project addresses the challenging task of detecting and classifying fallacious arguments in political debates. The main contributions are:

1. An extension of the ElecDeb60To16 dataset to include the 2020 Trump-Biden presidential debate, with updated token-level annotations of argumentative components, relations, and six categories of fallacious arguments.
2. A novel neural architecture, MultiFusion BERT, for fallacious argument detection and classification, combining text, argumentative features, and engineered features.

## Dataset: ElecDeb60to20

- Source: U.S. presidential election campaign debates (1960-2020)
- Annotations: Argumentative components (claims, premises), relations (support, attack), and six fallacy categories
- Fallacy categories: Ad Hominem, Appeal to Authority, Appeal to Emotion, False Cause, Slippery Slope, Slogans

## Methodology

- Task: Fallacious argument detection and classification (token-level)
- Model: MultiFusion BERT
- Features: Political discourse text, argumentative components, argumentative relations, Part-of-Speech tags

## Results

The following table shows the average macro F1 scores for fallacy detection using different models and features:

| Model | Features | Avg macro F1 Score |
|-------|----------|---------------------|
| BERT + LSTM | ❌ | 0.4697 |
| BERT + LSTM | Components and relations | 0.5142 |
| BERT + BiLSTM + LSTM | ❌ | 0.5495 |
| BERT + BiLSTM + LSTM | Components and relations | 0.5614 |
| BertFTC bert-base-uncased | ❌ | 0.7096 |
| BertFTC dbmdz/bert-large-cased-finetuned-conll03-english | ❌ | 0.7237 |
| DebertaFTC microsoft/deberta-base | ❌ | 0.7222 |
| ElectraFTC bhadresh-savani/electra-base-discriminator-finetuned-conll03-english | ❌ | 0.4033 |
| DistilbertFTC distilbert-base-cased | ❌ | 0.7010 |
| DistilbertFTC distilbert-base-uncased | ❌ | 0.7047 |
| MultiFusion BERT | Components, relations, and PoS | 0.7394 |

The best-performing model is MultiFusion BERT, which incorporates argumentative components, relations, and PoS features.

## Usage
The code runs under Python 3.9 or higher. The required packages are listed in the requirements.txt, which can be directly installed from the file:

```
pip install -r /path/to/requirements.txt
```

Our code is based on the transformer library version 4.28.0. See https://github.com/huggingface/transformers for more details.

## Citation

If you use this work, please cite:

```
@inproceedings{goffredo2023fallacies,
  title={Argument-based Detection and Classification of Fallacies in Political Debates},
  author={Goffredo, Pierpaolo and Chaves Espinoza, Mariana and Cabrio, Elena and Villata, Serena},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023}
}
```

## Contact

For any questions or issues, please open an issue in this repository or contact the authors.