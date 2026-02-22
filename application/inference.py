from .model import Gemma3Model
from .config import GEMMA3_CONFIG_270M
from .tokenizer import encode, decode
import torch

device = torch.device("cpu")

model = Gemma3Model(GEMMA3_CONFIG_270M)
model.load_state_dict(
    torch.load("checkpoints/best_model_params.pt", map_location=device)
)
model.to(device)
model.eval()

@torch.no_grad()
def generate(prompt, max_new_tokens=100):
    tokens = encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    output = model.generate(input_ids, max_new_tokens=max_new_tokens)

    return decode(output[0].tolist())