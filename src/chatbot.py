from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Chatbot:
    def __init__(self, model_name='distilgpt2', model_weights_path='model_weights.pth'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.load_model_weights(model_weights_path)

    def load_model_weights(self, model_weights_path):
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

    def generate_response(self, user_input):
        inputs = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    chatbot = Chatbot()
    user_input = input("You: ")
    response = chatbot.generate_response(user_input)
    print("Chatbot:", response)