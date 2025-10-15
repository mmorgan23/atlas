from transformers import AutoModelForCausalLM
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, model_name='distilgpt2'):
        super(Model, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate_response(self, input_ids, max_length=50):
        return self.model.generate(input_ids, max_length=max_length)