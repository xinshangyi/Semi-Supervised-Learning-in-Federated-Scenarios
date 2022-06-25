# Semi-Supervised Learning in Federated Scenarios

**Abstract:**  In this paper, a universal self-training paradigm for semi-supervised learning scenarios is proposed, which constructs the unsupervised loss term for unlabeled data combining core ideas of consistency regularization and entropy minimization. The paradigm uses two different data augmentations to accomplish artificial label generation and training processes simultaneously within one model. Based on the proposed self-training paradigm, we then explore semi-supervised learning in federated scenarios, which are divided into the standard and disjoint scenarios based on labeled data locations. In the standard scenario where labeled data are at clients, we propose Local Semi-Supervised Federated Learning to change local training ways from fully supervised to semi-supervised learning. In contrast to the performance of basic federated setting, we can speculate the cause of the negative influence of statistic heterogeneity on federated model learning and make local self-training a promising direction in high degree of Non-IID distributions. In the disjoint scenario where labeled data are at the server, we propose Global Distillation Self-Training method, which integrates global distillation loss and self-training loss into local training within federated learning framework, jointly with server-side fine-tuning to further stabilize and enhance the learning process of the global model. Our proposed method experimentally proves effective to train a global model distributedly in disjoint scenarios. 



## Parameters 

(some important from /utils/options.py)

| Parameter            | Description                                             |
| -------------------- | ------------------------------------------------------- |
| `frac`               | the fraction of parties to be sampled in each round.    |
| `dataset`            | Dataset to use. Options: `cifar10`. `cifar100`, `mnist` |
| `lr`                 | Learning rate.                                          |
| `bs`                 | Global batch size.                                      |
| `local_bs`           | Local training batch size.                              |
| `local_ep`           | Number of local epochs.                                 |
| `num_users`          | Number of parties.                                      |
| `global_aggregation` | Whether global_aggregation or local_aggregation         |
| `epochs`             | Number of communication rounds.                         |
| `iid`                | Whether IID or not.                                     |
| `noniid_type`        | Extreme noniid or dirichlet                             |
| `beta`               | Non-IID Dirichlet parameter.                            |
| `tempera`            | Distillation temperature for GDST.                      |
| `loss_type`          | Disjoint SSFL local loss type: GDST/ST/KD               |
| `gpu`                | Specify the device to run the program.                  |



## Usage 

IID: sampling in program execution 

Non-IID: extreme or dirichlet, program execution with 'pkl' file, sampling in advance by "/utils/sampling.py" 

### Standard SSFL Scenario

```python
python main_fed.py --global_aggregation --noniid_type 2 --beta 0.1
```

### Disjoint SSFL Scenario

```python
python disjoint_fed.py --noniid_type 2 --beta 0.1 --tempera 2 --loss_type GDST
```



## Citation

Please cite our paper if you find this code useful for your research.

```
@inproceedings{DBLP:conf/globecom/LiuZXJ021,
  author    = {Xinyi Liu and
               Linghui Zhu and
               Shu{-}Tao Xia and
               Yong Jiang and
               Xue Yang},
  title     = {{GDST:} Global Distillation Self-Training for Semi-Supervised Federated
               Learning},
  booktitle = {{IEEE} Global Communications Conference, {GLOBECOM} 2021, Madrid,
               Spain, December 7-11, 2021},
  pages     = {1--6},
  publisher = {{IEEE}},
  year      = {2021}
}
```

