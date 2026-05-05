# This is a modified version of the GraphRAG Framework "Digimon" that incorporates a hybridized version of RAPTOR and HippoRAG nicknamed "RAPPO"
## The original framework can be found here: https://github.com/JayLZhou/GraphRAG#
# Experiment Setup

This README explains how to install dependencies, activate the conda environment, and configure the experiment for your local LLM and embedding model.

## 1. Install Dependencies

Create a conda environment from `experiment.yml`:

```bash
conda env create -f experiment.yml -n <your_experiment_name>
```

Replace `<your_experiment_name>` with the name you want to use for the environment.

## 2. Activate the Conda Environment

```bash
conda activate <your_experiment_name>
```

## 3. Configure the LLM and Embedding Model

Modify `config2.yaml` to match your local LLM server, embedding model, dataset path, and hardware setup.

For example, on a machine using `llama3.1:8b` through Ollama, the configuration may look like this:

```yaml
llm:
  api_type: "open_llm" # or openai
  base_url: "http://127.0.0.1:11434/v1"
  model: "llama3.1:8b"
  api_key: "ollama"

embedding:
  api_type: "hf" # or ollama / openai
  base_url: ""
  api_key: ""
  model: "BAAI/bge-m3"
  cache_folder: "/home/User/.cache/huggingface"
  device: "cuda:1"
  dimensions: 1024
  embed_batch_size: 128

data_root: "/home/User/GraphRAG_test/Data" # Root directory for data

working_dir: ./ # Result directory for the experiment
exp_name: run_experiment # Experiment name
```

## Machine-Specific Fields

The following fields are machine-specific and may need to be changed before running the experiment:

| Field | Description |
|---|---|
| `base_url` | URL for your LLM server |
| `model` | LLM model name available through your server |
| `cache_folder` | Local HuggingFace cache directory |
| `device` | GPU device to use for embeddings, for example `cuda:0` |
| `data_root` | Path to the dataset directory on your machine |

## 5. Run a Specific Method

```bash
python main.py -opt Option/Method/<METHOD>.yaml -dataset_name your_dataset
```
For example, 
```bash
python main.py -opt Option/Method/RAPPO.yaml -dataset_name multihop-rag
```


(See the original Github linked above for more detailed instructions)
-- Note: The artifacts folder is unnecessary to reproducing the experiment, and they simply show past logs of experiments.
