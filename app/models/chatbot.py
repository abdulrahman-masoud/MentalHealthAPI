from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class MentalHealthChatbot:
    def __init__(self, model_name="tanusrich/Mental_Health_Chatbot"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print(f"Loading model on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.to(self.device)
        print("Model loaded successfully!")
    
    def generate_response(self, message, max_tokens=200, temperature=0.7, 
                          top_k=50, top_p=0.9, repetition_penalty=1.2):
        inputs = self.tokenizer(message, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    
    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "parameters": self.model.num_parameters()
        }