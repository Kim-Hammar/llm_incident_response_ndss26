from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if __name__ == '__main__':
    print("Loading the fine-tuned incident response LLM.")
    model = AutoModelForCausalLM.from_pretrained(
        "kimhammar/LLMIncidentResponse",
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("kimhammar/LLMIncidentResponse")
    print(f"LLM loaded successfully on device: {model.device}")
