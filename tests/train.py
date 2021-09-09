from PIL import Image
import clip
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup
import time

model_name = "./models/ViT-B-16.pt"
jit = True
#  batch size of 32,768
batch_size = 64
print("model_name: ",model_name,", jit: ", jit, ", batch_size:", batch_size)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
if jit:
    model, preprocess = clip.load(model_name, device=device, jit=True)
else:
    model, preprocess = clip.load(model_name, device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).repeat(batch_size,1,1,1).to(device)
text = clip.tokenize(["a diagram"]*batch_size).to(device)
labels = torch.Tensor(np.arange(batch_size)).long().to(device)

# a fixed temperature of 0.07
# Adam optimizer with decay and cosine schedule
# Warm-up iterations 2000
# Mixed-precision
# ViT lr=1e-6, RN lr=1e-8, Adam_beta=[0.9,0.999(RN)/0.98(ViT)]
# Training epochs 32
epochs = 32
warmup_steps = 2000
num_steps = 6100 # 400m / 2 / batch size 32,768
if 'RN' in model_name:
    lr = 1e-8
    adam_beta = [0.9, 0.999]
else:
    lr = 1e-6
    adam_beta = [0.9, 0.98]
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=adam_beta, weight_decay=0.2)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps, last_epoch=-1)

loss_fnc = torch.nn.CrossEntropyLoss()

test_steps = 100

step_count = 0
time_start=time.time()
model.train()
while step_count<test_steps:
    logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print(logits_per_image.shape, logits_per_text.shape, probs.shape)
    # print(logits_per_image, logits_per_text, probs)
    loss_i = loss_fnc(logits_per_image, labels)
    loss_t = loss_fnc(logits_per_text, labels)
    loss = ((loss_i + loss_t)/2.0).mean()
    # print(loss_i, loss_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step_count % (test_steps//10) == 0:
        time_end=time.time()
        during = time_end-time_start
        print(step_count//(test_steps//100), "%, ", 'time cost',during,'s ,', "throughput is ", (test_steps//10)*batch_size//during)
        time_start=time.time()
    step_count += 1

print("model_name: ",model_name,", jit: ", jit, ", batch_size:", batch_size)