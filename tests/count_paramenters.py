import clip
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

device = "cpu"
for model_name in clip.available_models():
    model, _ = clip.load(model_name, device=device)
    print(model_name, get_parameter_number(model))