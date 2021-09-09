import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("./data/cifar100"), download=True, train=True)

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)


t_count = 0
f_count = 0
# Calculate features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    all_count = len(cifar100)
    for i in range(all_count):
    # i = 3637
    # if i == 3637:
    # Prepare the inputs
        image, class_id = cifar100[i]
        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)
        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = model.logit_scale.exp() * image_features @ text_features.t()
        _, indices = similarity.softmax(dim=-1).topk(1)
        # print(indices,class_id, indices == class_id)
        # exit()
        if indices == class_id:
            t_count += 1
        else:
            f_count += 1
        if i % 1000 == 0:
            print(i)
        
print(t_count/(all_count), t_count, f_count)

    