import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model_name_or_path="IPEC-COMMUNITY/spatialvla-4b-224-pt"
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()

image = Image.open("example.png").convert("RGB")
prompt = "What action should the robot take to pick the cup?"
inputs = processor(images=[image], text=prompt, return_tensors="pt")
generation_outputs = model.predict_action(inputs)

actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")
print(actions)