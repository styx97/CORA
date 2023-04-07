set -e

env_name="${1:-cora_mdpr}"

if ! command -v conda &>/dev/null; then
    echo "'conda' not found; install it here: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if ! { conda env list | grep "${env_name}"; } >/dev/null 2>&1; then
    conda create -y -n "${env_name}" python=3.8
fi

eval "$(conda shell.bash hook)"
conda activate ${env_name}



# huggingface installs
pip install --no-cache transformers==3.0.2 

# Torch installs
pip install --no-cache torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# NLP tools
pip install --no-cache spacy nltk 

# dpr stuff 
pip install --no-cache faiss-cpu

# packages for running jobs
pip install submitit 

# misc other standard-ish packages
pip install --no-cache numpy pandas scipy einops scikit-learn matplotlib jupyterlab wandb

pip install --no-cache rich loguru jsonlines
