# Exploiting Cross-Modal Prediction and Relation Consistency for Semi-Supervised Image Captioning

This repository includes the implementation for Exploiting Cross-Modal Prediction and Relation Consistency for Semi-Supervised Image Captioning.

## Requirements

- Python 3.6
- Java 1.8.0
- PyTorch 1.0
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule)
- tensorboardX


## Training CPRC

### Prepare data

See details in `data/README.md`.

(**notes:** Set `word_count_threshold` in `scripts/prepro_labels.py` to 4 to generate a vocabulary of size 10,369.)

You should also preprocess the dataset and get the cache for calculating cider score for [SCST](https://arxiv.org/abs/1612.00563):

```bash
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```
### Start training

```bash
$ CUDA_VISIBLE_DEVICES=0 sh train.sh
```

See `opts.py` for the options.


### Evaluation

```bash
$ CUDA_VISIBLE_DEVICES=0 python eval.py --model log/log_aoanet_rl/model.pth --infos_path log/log_aoanet_rl/infos_aoanet.pkl  --dump_images 0 --dump_json 1 --num_images -1 --language_eval 1 --beam_size 2 --batch_size 100 --split test
```


## Reference

If you find this repo helpful, please consider citing:

```
@article{yang2022,
  title={Exploiting Cross-Modal Prediction and Relation Consistency for Semi-Supervised Image Captioning},
  author={Yang Yang, Hong-Chen Wei, Heng-Shu Zhu, Dian-Hai Yu, Hui Xiong, Jian Yang},
  booktitle={IEEE Transactions on Cybernetics},
  year={2022}
}
```

## Acknowledgements

This repository is based on [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch), and you may refer to it for more details about the code.
