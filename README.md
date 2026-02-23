# ProtegoFed: Backdoor-Free Federated Instruction Tuning with Interspersed Poisoned Data

## Environment Setup

### Create Virtual Environment

```bash
# Create conda environment
conda create -n protegofed python=3.10.16  
conda activate protegofed
```
### Install Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt
```


* Training Data. We provide the raw datasets in [./datasets/QuestionAnswering](./datasets/QuestionAnswering/).


## Usage

```bash
    python xxx.py --config_path=./genConfigs/xxx.json --poisoner=xxx
```

## Acknowledgement

This work could not have been completed without the help of the following repositories:

- OpenBackdoor: https://github.com/thunlp/OpenBackdoor
- PEFT: https://github.com/huggingface/peft
- GraCeFul: https://github.com/ZrW00/GraceFul