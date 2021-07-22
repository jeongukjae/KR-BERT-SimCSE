# KR-BERT-SimCSE

This repository contains codes for SimCSE using TensorFlow 2 and KR-BERT.

## Training

### Unsupervised

```
python train_unsupervised.py --mixed_precision
```

I used [Korean Wikipedia Corpus](https://github.com/jeongukjae/korean-wikipedia-corpus) that is divided into sentences in advance. (Check out [tfds-korean catalog page](https://jeongukjae.github.io/tfds-korean/datasets/korean_wikipedia_corpus.html) for details)

* Settings
    * peak learning rate 3e-5
    * batch size 64
    * Total steps: 25,000
    * 0.05 warmup rate, and linear decay learning rate scheduler
    * temperature 0.05
    * evalaute on KLUE STS and KorSTS every 250 steps
    * max sequence length 64
    * Use pooled outputs for training, and `[CLS]` token's representations for inference

The hyperparameters were not tuned and mostly followed the values in the paper.

### Supervised

// TODO

## Results

### KorSTS (dev set results)

|model|||Spearman R|
|---|---|---|--:|
|KR-BERT base SimCSE|unsupervised                           |bi encoding|79.99|
|||||
|SRoBERTa base*     |unsupervised                           |bi encoding|63.34|
|SRoBERTa base*     |unsupervised, trained on KorNLI        |bi encoding|76.48|
|SRoBERTa base*     |supervised                             |bi encoding|83.68|
|SRoBERTa base*     |supervised, trained on KorNLI -> KorSTS|bi encoding|83.54|
|SRoBERTa large*    |supervised                             |bi encoding|84.74|

* *: results from [Ham et al., 2020](https://arxiv.org/abs/2004.03289).

### KorSTS (test set results)

|model|||Spearman R|
|---|---|---|--:|
|KR-BERT base SimCSE|unsupervised                           |bi encoding   |73.25|
|||||
|SRoBERTa base*     |unsupervised                           |bi encoding   |48.96|
|SRoBERTa base*     |unsupervised, trained on KorNLI        |bi encoding   |74.19|
|SRoBERTa base*     |supervised                             |bi encoding   |78.94|
|SRoBERTa base*     |supervised, trained on KorNLI -> KorSTS|bi encoding   |80.29|
|SRoBERTa large*    |supervised                             |bi encoding   |79.55|
|SRoBERTa base*     |supervised                             |cross encoding|83.00|
|SRoBERTa large*    |supervised                             |cross encoding|85.27|

* *: results from [Ham et al., 2020](https://arxiv.org/abs/2004.03289).

### KLUE STS (dev set results)

|model|||Pearson R|
|---|---|---|--:|
|KR-BERT base SimCSE|unsupervised|bi encoding   |74.45|
|||||
|KR-BERT base*      |supervised  |cross encoding|87.50|

* *: results from [Park et al., 2021](https://arxiv.org/abs/2105.09680).

## References

```
@misc{gao2021simcse,
    title={SimCSE: Simple Contrastive Learning of Sentence Embeddings},
    author={Tianyu Gao and Xingcheng Yao and Danqi Chen},
    year={2021},
    eprint={2104.08821},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

```
@misc{park2021klue,
    title={KLUE: Korean Language Understanding Evaluation},
    author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
    year={2021},
    eprint={2105.09680},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

```
@misc{ham2020kornli,
    title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
    author={Jiyeon Ham and Yo Joong Choe and Kyubyong Park and Ilji Choi and Hyungjoon Soh},
    year={2020},
    eprint={2004.03289},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
