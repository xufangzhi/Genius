<h1 align="center">
<img src="./assets/logo.png" width="100" alt="Symbol-LLM" />
<br>
Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning
</h1>



<p align="center">
  <a href="https://xufangzhi.github.io/symbol-llm-page/"><b>[🌐 Website]</b></a> •
  <a href="https://arxiv.org/abs/2311.09278"><b>[📜 Paper]</b></a> •
  <a href="https://huggingface.co/Symbol-LLM/Symbol-LLM-7B-Instruct"><b>[🤗 HF Models]</b></a> •
  <a href="https://huggingface.co/datasets/Symbol-LLM/Symbolic_Collection"><b>[🤗 HF Dataset]</b></a> •
  <a href="https://github.com/xufangzhi/Symbol-LLM"><b>[🐱 GitHub]</b></a>
  
</p>


<p align="center">
Repo for "<a href="https://arxiv.org/abs/2311.09278" target="_blank">Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning</a>"
</p>


## 🔥 News

- [2025/02/16] 🔥🔥🔥 Genius is submitted to ACL ARR Feb. 2025 !


## 🚀 Quick Start

To try on Symbol-LLM, please use the Transformer library:

```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Symbol-LLM/Symbol-LLM-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Symbol-LLM/Symbol-LLM-7B-Instruct")
```


To utilize our symbolic collection, please load the dataset:

```python
from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("Symbol-LLM/Symbolic_Collection")
```

## 📃 Deployed As A WebUI
The implementation of WebUI is modified from [text-generation-webui](https://github.com/oobabooga/text-generation-webui). The running script is as follows:

```bash
cd demo-webui/
python server.py --model <model_name> --api --share --gpu-memory 40 40 --compute_dtype float32 --bf16
```


## 📒 Note
This work is still under review. We will open-source the model weights, symbolic collection and the code.


## 🔧 Repo Structure
This repo contains the training scripts and the demo deployment. Detailed structure is as follow:
```
.
├── README.md
├── logo.png
├── demo-webui
```

## Citation
If you find it helpful, please kindly cite the paper.
```
@article{xu2023symbol,
  title={Symbol-LLM: Towards Foundational Symbol-centric Interface For Large Language Models},
  author={Xu, Fangzhi and Wu, Zhiyong and Sun, Qiushi and Ren, Siyu and Yuan, Fei and Yuan, Shuai and Lin, Qika and Qiao, Yu and Liu, Jun},
  journal={arXiv preprint arXiv:2311.09278},
  year={2023}
}
```
