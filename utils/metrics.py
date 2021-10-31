# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from collections import deque
import time
import torch
import horovod.torch as hvd

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_accuracy(predictions, labels):
    _, ind = torch.max(predictions.float(), 1)
    # provide labels only for samples, where prediction is available (during the training, not every samples prediction is returned for efficiency reasons)
    labels = labels[-predictions.size()[0]:]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / labels.size()[0]
    return accuracy


class Metrics:
    def __init__(self, running_mean_length=100, distributed=False):
        self.storage = {}
        self.running_mean_length = running_mean_length
        self.distributed = distributed
        self.last_time = time.time()


    def save_value(self, name, value):
        if name in self.storage.keys():
            self.storage[name]["sum_item"] += value
            self.storage[name]["count_item"] += 1
            self.storage[name]["sum_running_mean"] += value
            if len(self.storage[name]["running_mean_items"]) == self.running_mean_length:
                self.storage[name]["sum_running_mean"] -= self.storage[name]["running_mean_items"].popleft()
            self.storage[name]["running_mean_items"].append(value)
        else:
            entry = {"sum_item": value,
                     "count_item": 1,
                     "sum_running_mean": value,
                     "running_mean_items": deque([value], maxlen=self.running_mean_length)}
            self.storage[name] = entry


    def get_running_mean(self, name):
        running_sum = self.storage[name]["sum_running_mean"]
        running_count = len(self.storage[name]["running_mean_items"])
        if self.distributed:
            running_sum = sync_metrics(running_sum)
            running_count = sync_metrics(running_count)
        return running_sum / running_count


    def get_value(self, name):
        sum_value = self.storage[name]["sum_item"]
        count_value = self.storage[name]["count_item"]
        self.storage[name]["sum_item"] = 0.0
        self.storage[name]["count_item"] = 0
        if self.distributed:
            sum_value = sync_metrics(sum_value)
            count_value = sync_metrics(count_value)
        return sum_value / count_value


    def get_count(self):
        count = next(iter(self.storage.values()))["count_item"]
        if self.distributed:
            count = sync_metrics(count, average=False)
        return count


    def get_elapsed_time(self):
        now = time.time()
        elapsed_time = now-self.last_time
        if self.distributed:
            elapsed_time = sync_metrics(elapsed_time)
        self.last_time = time.time()
        return elapsed_time


def sync_metrics(value, average=True):
    tensor = torch.Tensor([value])
    avg_value = hvd.allreduce(tensor, average=average)
    return float(avg_value.item())
