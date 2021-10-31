import urllib
from urllib import error
from urllib.request import urlopen
from urllib.error import URLError
import os
from tqdm.autonotebook import tqdm
import threading
import queue
import random
import string


image_path = "/localdata/minghongc/workspace/datasets/GCC/images"
caption_path = "/localdata/minghongc/workspace/datasets/GCC"

images = []
captions = []
index = 0


def read_file():
    f1 = open("/localdata/minghongc/datasets/GCC/Train-GCC-training.tsv", "r", encoding="utf-8")
    f1_lines = f1.readlines()
    f1.close()

    f2 = open("/localdata/minghongc/datasets/GCC/Validation-GCC-1.1.0-Validation.tsv", "r", encoding="utf-8")
    f2_lines = f2.readlines()
    f2.close()

    f_lines = f1_lines + f2_lines

    f_queue = queue.Queue()
    for line in tqdm(f_lines):
        cap, img = line.strip().split("\t")
        f_queue.put((cap, img))

    return f_queue


def generate_name():
    random_str = ''.join(random.sample(string.ascii_letters + string.digits, 20))

    return random_str + ".jpg"


def craw(img_url):
    try:
        img = urlopen(img_url).read()
        return img

    except Exception as e:
        return None


def writer(cap, img_name, img, fout):
    # write the pair 
    img_w = open(f"{image_path}/{img_name}", "wb")
    img_w.write(img)
    img_w.close()

    fout.write(cap + "\t" + img_name + "\n")


def do_craw(line_queue, ele_queue):
    while True:
        cap, img_url = line_queue.get()
        img = craw(img_url)
        img_name = generate_name()
        if img:
            ele_queue.put((cap, img_name, img))

        
def do_writer(ele_queue, fout):
    while True:
        cap, img_name, img = ele_queue.get()
        print("The size of ele_queue: ", ele_queue.qsize())
        writer(cap, img_name, img, fout)

  
if __name__ == "__main__":
    line_queue = read_file()  # [(cap, img), ...]
    ele_queue = queue.Queue()

    for idx in range(1200):
        t = threading.Thread(target=do_craw, args=(line_queue, ele_queue))
        t.start()

    fout = open("cap_img.txt", "w", encoding="utf-8")

    for idx in range(1500):
        t = threading.Thread(target=do_writer, args=(ele_queue, fout))
        t.start()

    # fout.close()
    # print("Done")
