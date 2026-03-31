# MMORF

## Setup

### Manual Installation

1. Clone this repository and change directories to the project root.
```
git clone https://github.com/ninglab/MMORF.git
cd MMORF
```

2. Download and unpack necessary data/model files from this link:
[https://frazierbaker.com/mmorf_largefiles.zip](https://frazierbaker.com/mmorf_largefiles.zip)

Unzip the files and copy them into in `mmorf` so that they are in the following locations:
```
saved_model/policy_model.ckpt
saved_model/template_rules.dat
saved_model/value_pc.pt
prepare_data/origin_dict.csv
prepare_data/purch_cost.json
```

3. Install Python 3.11.1, then run the following commands from the project root:
```
pip install -e .
cd mmorf && install -e rdchiral && cd ..
```

4. Setup environment variables in `.env`.  To use Claude, you will need to specify an Anthropic API Key.
To use Qwen30B or Mistral Nemo 12B, you will need an OpenAI-compatible vLLM server running separately with the desired model,
and you must specify the hostname, port, and API key in the `.env` file.
You may also set a custom dataset here by changing the `DATASET_PREFIX`.  Refer to [Dataset](#dataset) for more information on creating a custom dataset.

```
ANTHROPIC_API_KEY=sk-ant-api00000000000000000
VLLM_API_KEY=...
VLLM_PORT=...
VLLM_URL=...
DATASET_PREFIX=all
```

*For more information on running a vLLM server, please see their [documentation](https://docs.vllm.ai/en/latest/).*

## Docker Installation

Alternatively, you may opt to download pre-built Docker image.  In this case, you do not need to manually install MMORF.  Instead, you can follow the instructions below:

1. Setup environment variables in `.env`.  To use Claude, you will need to specify an Anthropic API Key.
To use Qwen30B or Mistral Nemo 12B, you will need an OpenAI-compatible vLLM server running separately with the desired model,
and you must specify the hostname, port, and API key in the `.env` file.
You may also set a custom dataset here by changing the `DATASET_PREFIX`.  Refer to [Dataset](#dataset) for more information on creating a custom dataset.

```
ANTHROPIC_API_KEY=sk-ant-api00000000000000000
VLLM_API_KEY=...
VLLM_PORT=...
VLLM_URL=...
DATASET_PREFIX=all
```

*For more information on running a vLLM server, please see their [documentation](https://docs.vllm.ai/en/latest/).*


2. Run the Docker container with `.env` mounted as a volume.
```
docker run -it -v .env:/app/.env frazierbaker/mmorf
```
This will provide you with a command-line interface to explore MMORF systems.


## Running MMORF Representative Systems<a name="running"></a>

### Running MASIL

To run MASIL on one of the tasks in our dataset, use the following command:

```
bash scripts/test_masil.sh {BaseLLM} {OneIndex}
```

Where `{BaseLLM}` is one of the following: `Claude`, `Qwen30B` or `Mistral` and `{OneIndex}` is the line number corresponding to the product in `data/all_products.txt` and the instructions in `data/all_tasks.txt`.

To run the entire benchmark, simply repeat this call for all possible indices:
```
for idx in $(seq 1 218); do 
    bash scripts/test_masil.sh {BaseLLM} $idx
done
```

Output will be stored in a new directory, `MASIL_all_{BaseLLM}`.


### Running RFAS

To run RFAS on one of the tasks in our dataset, use the following command:

```
bash scripts/test_rfas.sh {BaseLLM} {OneIndex}
```

Where `{BaseLLM}` is one of the following: `Claude`, `Qwen30B` or `Mistral` and `{OneIndex}` is the line number corresponding to the product in `data/all_products.txt` and the instructions in `data/all_tasks.txt`.

To run the entire benchmark, simply repeat this call for all possible indices:
```
for idx in $(seq 1 218); do 
    bash scripts/test_rfas.sh {BaseLLM} $idx
done
```

Output will be stored in a new directory, `RFAS_all_{BaseLLM}`.

## Dataset<a name="dataset"></a>

The benchmark dataset is stored in `data` across two text files:
 - `all_products.txt`, which contains the product SMILES, one per line and
 - `all_tasks.txt`, which contains the task instructions, one per line.

The lines of these files are associated by line number.  To test your own data, you may add similar files named `data/{prefix}_products.txt` and `data/{prefix}_tasks.txt` and changing `DATASET_PREFIX={prefix}` in the `.env`.