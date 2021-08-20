from clip_ipu.model_ipu import PipelinedWithLossForCLIP
import torch
import clip
from PIL import Image
import time
import poptorch

# custom
model_name = "./models/ViT-B-32.pt"
jit = False
batch_size = 1
print("model_name: ",model_name,", jit: ", jit, ", batch_size:", batch_size)

device = "cpu"

model = PipelinedWithLossForCLIP(model_name, jit=jit)
model = model.half()
preprocess = model.preprocess

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

all = 1000
if batch_size>1:
    all = 100
    image = image.repeat(batch_size,1,1,1).to(device)

# print("image size:",image.shape, "text size:", text.shape)
model.train()
opts = poptorch.Options()
poptorch_model = poptorch.inferenceModel(model.eval(), options=opts,)
with torch.no_grad():
    count = 1
    time_start=time.time()
    
    while count<all:
        # image_features = model.encode_image(image)
        # text_features = model.encode_text(text)
    
        probs = poptorch_model(image, text)
        if count % (all//10) == 0:
            time_end=time.time()
            during = time_end-time_start
            print(count//(all//100), "%, ", 'time cost',during,'s ,', "throughput is ", (all//10)*batch_size//during)
            time_start=time.time()
        count += 1
    
print("model_name: ",model_name,", jit: ", jit, ", batch_size:", batch_size)
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]




