
from clip_ipu.model_ipu import PipelinedWithLossForCLIP
from PIL import Image
import clip
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup
import time
import poptorch

model_name = "./models/ViT-B-32.pt"
jit = True
#  batch size of 32,768
batch_size = 2


device = "cpu"

model = PipelinedWithLossForCLIP(model_name, jit=jit)
model.half()
preprocess = model.preprocess

images = preprocess(Image.open("CLIP.png")).unsqueeze(0).repeat(batch_size,1,1,1)
texts = clip.tokenize(["a diagram"]*batch_size)
labels = torch.Tensor(np.arange(batch_size)).long()

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
    adam_beta = (0.9, 0.999)
else:
    lr = 1e-6
    adam_beta = (0.9, 0.98)

# # Do not apply weight_decay for one-dimensional parameters
# regularized_params = []
# non_regularized_params = []
# for param in model.parameters():
#     if param.requires_grad:
#         if len(param.shape) == 1:
#             non_regularized_params.append(param)
#         else:
#             regularized_params.append(param)

# params = [
#     {"params": regularized_params, "weight_decay": 0.2},
#     {"params": non_regularized_params, "weight_decay": 0}
# ]
params = model.parameters()
optimizer = poptorch.optim.AdamW(params, lr=lr, betas=adam_beta, weight_decay=0.2, accum_type=torch.float16) 
# optimizer = poptorch.optim.SGD(params, lr=lr, weight_decay=0.2, use_combined_accum=True)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps, last_epoch=-1)

test_steps = 100

step_count = 0
time_start=time.time()

model.train()
opts = poptorch.Options()
poptorch_model = poptorch.trainingModel(model,
                                        options=opts,
                                        optimizer=optimizer)
poptorch_model.setOptimizer(optimizer)

poptorch_model.compile(images, texts, labels)
exit()
while step_count<test_steps:
    
    _, loss = poptorch_model(images, texts, labels)
    scheduler.step()
    poptorch_model.setOptimizer(optimizer)

    if step_count % (test_steps//10) == 0:
        time_end=time.time()
        during = time_end-time_start
        print(step_count//(test_steps//100), "%, ", 'time cost',during,'s ,', "throughput is ", (test_steps//10)*batch_size//during)
        time_start=time.time()
    step_count += 1
