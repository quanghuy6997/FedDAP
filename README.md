## FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift [CVPR 2026]

Paper: https://arxiv.org/abs/2604.06795

## Abstract

Federated Learning (FL) enables decentralized model training across multiple clients without exposing private data, making it ideal for privacy-sensitive applications. However, in real-world FL scenarios, clients often hold data from distinct domains, leading to severe domain shift and degraded global model performance. To address this, prototype learning has been emerged as a promising solution, which leverages class-wise feature representations. Yet, existing methods face two key limitations: (1) Existing prototype-based FL methods typically construct a $\textit{single global prototype}$ per class by aggregating local prototypes from all clients without preserving domain information.  (2) Current feature-prototype alignment is $\textit{domain-agnostic}$, forcing clients to align with global prototypes regardless of domain origin. To address these challenges, we propose Federated Domain-Aware Prototypes  (FedDAP) to construct domain-specific global prototypes by aggregating local client prototypes within the same domain using a similarity-weighted fusion mechanism. These global domain-specific prototypes are then used to guide local training by aligning local features with prototypes from the same domain, while encouraging separation from prototypes of different domains. This dual alignment enhances domain-specific learning at the local level and enables the global model to generalize across diverse domains. Finally, we conduct extensive experiments on three different datasets: DomainNet, Office-10, and PACS to demonstrate the effectiveness of our proposed framework to address the domain shift challenges.

Recommended environment:

- Python 3.9+
- PyTorch
- torchvision
- numpy
- scipy
- scikit-learn
- pillow
- tqdm
- setproctitle

Install the main dependencies with:

```bash
pip install torch torchvision numpy scipy scikit-learn pillow tqdm setproctitle
```

## Dataset Layout

The default dataset root in this project is:

```text
./datasets/pic_cls/
```

Expected folders used by the code include:

```text
datasets/pic_cls/
	domain/
	Office_Caltech_10/
	PACS/
```

Supported datasets in the current codebase:

- `fl_digits`
- `fl_officecaltech`
- `fl_pacs`
- `fl_domainnet`

## How to Run Experiment

### Main entry

Run experiments from the project root:

```bash
python main.py --model feddap --dataset fl_officecaltech
```

The script automatically loads dataset-model specific default hyperparameters from `utils/best_args.py`.

### Important arguments

- `--model`: training method, such as `feddap`, `fedavg`, `fedplvm`, `feddgga`, or `fpl`
- `--dataset`: one of `fl_digits`, `fl_officecaltech`, `fl_pacs`, `fl_domainnet`
- `--communication_epoch`: number of federated communication rounds
- `--local_epoch`: number of local epochs per client in each round
- `--parti_num`: number of clients
- `--device_id`: CUDA device id
- `--seed`: random seed

### Example commands

FedDAP on Office-Caltech:

```bash
python main.py --model feddap --dataset fl_officecaltech --communication_epoch 100 --local_epoch 10 --parti_num 10 --device_id 0 --seed 0
```


FedDAP on PACS:

```bash
python main.py --model feddap --dataset fl_pacs  --communication_epoch 100 --local_epoch 10 --parti_num 12 --device_id 0 --seed 0
```


## Citation

If you use this repository in your research, please cite the paper:

```bibtex
@InProceedings{Le_2026_CVPR,
    author    = {Le, Huy Q. and Nguyen, Loc X. and Qiao, Yu and Kim, Seong Tae and Huh, Eui-Nam and Hong, Choong Seon},
    title     = {FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {3390-3399}
}
```

If the final published version has different page numbers or proceedings metadata, please update the BibTeX entry accordingly.
