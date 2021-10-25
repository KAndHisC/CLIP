import os
import clip
import torch
from torchvision.datasets import CIFAR100
import numpy as np

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("./data/cifar100"), download=True, train=True)

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)


r1_count = 0
r5_count = 0
a_value = []
# Calculate features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    all_count = len(cifar100)
    print(all_count)
    for i in range(all_count):
        # i = 3637
        # i = 1
        # Prepare the inputs
        image, class_id = cifar100[i]
        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)
        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = (100 * image_features @ text_features.t()).softmax(dim=-1)
        value, recall_at_1 = similarity[0].topk(1)
        values, recall_at_5 = similarity.softmax(dim=-1)[0].topk(5)
        value = similarity.softmax(dim=-1)[0][class_id]
        # print(value)

        # print(indices,class_id, indices == class_id)
        # exit()
        if recall_at_1 == class_id:
            r1_count += 1
            r5_count += 1
            a_value.append(value.cpu().numpy())
        elif class_id in recall_at_5:
            r5_count += 1
        
        if i % 1000 == 0:
            print(100*i/all_count, "%", r1_count/(i+1), r5_count/(i+1), np.array(a_value).mean())
        
print(r1_count/all_count, r5_count/all_count)
print(np.array(a_value).mean())

    