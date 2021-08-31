from clip_ipu import SyntheticData
from clip_ipu.model_ipu import PipelinedWithLossForCLIP
import torch
import clip
from PIL import Image
import time
import poptorch

# custom
model_name = "./models/ViT-B-32.pt"
bs = 64
iteration = 64
batch_size = bs*iteration
print("model_name: ", model_name, " batch_size:", batch_size)

model = PipelinedWithLossForCLIP(model_name)
preprocess = model.preprocess


device = "cpu"
# images = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
image = preprocess(Image.open("CLIP.png")).to(device)
texts = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# if batch_size>1:
#     images = images.repeat(batch_size,1,1,1).to(device)

# print("image size:",image.shape, "text size:", text.shape)
model.eval()
model.setTextAsCategories(texts)
model = model.half()

# setting IPU
opts = poptorch.Options()
opts.deviceIterations(iteration)
opts.setAvailableMemoryProportion({"IPU0":0.5, "IPU1":0.7, "IPU2":0.7, "IPU3":0.7})

dataset = SyntheticData(image)
dataloader = poptorch.DataLoader(opts, dataset, batch_size=bs, shuffle=False, num_workers=16)


# testing
poptorch_model = poptorch.inferenceModel(model.eval(), options=opts)

with torch.no_grad():
    time_start=time.time()

    for batch_number, images in enumerate(dataloader):
        probs = poptorch_model(images)
        if batch_number % 10 == 0:
            time_end=time.time()
            during = time_end-time_start
            print(batch_number, "%, ", 'time cost',during,'s ,', "throughput is ", 10*batch_size//during)
            time_start=time.time()
            if batch_number == 100:
                break
    
print("model_name: ", model_name, ", batch_size:", batch_size)
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]




