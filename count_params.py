import timm
from tqdm import tqdm

model_names = timm.list_models(pretrained=True)
list_parameters = {}
for model in tqdm(model_names):
    m = timm.create_model(model, pretrained=False, in_chans=3)
    list_parameters[model] = sum(p.numel() for p in m.parameters() if p.requires_grad)


max_value = max(list_parameters.values())  # maximum value
max_keys = [k for k, v in list_parameters.items() if v == max_value]
print(max_keys)
