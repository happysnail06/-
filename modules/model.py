import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

class ModelHandler:
    def __init__(self):
        model_name = "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
        # model_path = snapshot_download(repo_id=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)