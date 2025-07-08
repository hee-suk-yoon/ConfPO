# ConfPO


## Installation
```bash
# Clone this repo
git clone https://github.com/hee-suk-yoon/ConfPO
cd ConfPo

# Create a conda enviroment
1. conda env create --name confpo python=3.11
2. conda activate confpo
3. pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
4. conda env update --file environment.yml --prune
```

## Running Experiments
1. SimPO (Baseline)
```bash
cd trl-main
bash commands/run_simpo.sh
```

2. ConfPO (Ours)
```bash
cd trl-main
bash commands/run_confpo.sh
```

