import torch
from torch.utils.data import Dataset
import json
import string
import random

# used to make large embeddings feasible by exporting some to disk
class Embedding(Dataset):
    def __init__(self, data_path, axis=0):
        self.identifier = f"[{''.join(random.choices(string.ascii_letters + string.digits, k=5))}]"
        self.batch_file_paths = []
        self.batch_idxs = []
        self.data_path = data_path
        self.last_loaded_batch = None
        self.last_loaded_batch_edges = (-1, -1)
        self.shape = None
        self.axis = axis

    def assemble(self):
        tensors = []
        for batch_file_path in self.batch_file_paths:
            tensors.append(torch.load(batch_file_path))
        return torch.cat(tensors, dim=self.axis)

    def add(self, batch_tensor): # add new embedding batch at the END of the overall embedding list
        batch_tensor_path = self.data_path+self.identifier+f"batch_{len(self.batch_file_paths)}.pt"
        self.batch_file_paths.append(batch_tensor_path)
        torch.save(batch_tensor, batch_tensor_path)
        if len(self.batch_idxs) > 0:
            self.batch_idxs.append(self.batch_idxs[-1]+len(batch_tensor))
        else:
            self.batch_idxs.append(len(batch_tensor))
        self.shape = tuple([self.batch_idxs[-1]] + list(batch_tensor.shape)[1:])
        
    def map_over(self, function, data_path=None, result_axis=0):
        if data_path is None:
            data_path = self.data_path
        result_embedding = Embedding(data_path,result_axis)
        for batch_file_path in self.batch_file_paths:
            batch_tensor = torch.load(batch_file_path)
            batch_result = function(batch_tensor)
            result_embedding.add(batch_result)
        return result_embedding
    
    def get_batch_tensor(self, idx):
        return torch.load(self.batch_file_paths[idx])

    def __len__(self):
        if len(self.batch_idxs) > 0:
            return self.batch_idxs[-1]
        else:
            return 0
    
    def __getitem__(self, idx):
        if idx <= self.last_loaded_batch_edges[0] or idx > self.last_loaded_batch_edges[1]:
            file_idx = 0
            file_start_offset = 0
            while idx > self.batch_idxs[file_idx]:
                file_start_offset = self.batch_idxs[file_idx]
                file_idx += 1
                if file_idx >= len(self.batch_idxs):
                    raise IndexError()
            self.last_loaded_batch = torch.load(self.batch_file_paths[file_idx])
            self.last_loaded_batch_edges = (file_start_offset, self.batch_idxs[file_idx])
            
        batch_tensor = self.last_loaded_batch
        local_idx = idx - self.last_loaded_batch_edges[0]
        return batch_tensor[local_idx]
    
    def save(self):
        save_dict = {
            "file_paths": self.batch_file_paths,
            "batch_idxs": self.batch_idxs,
            "identifier": self.identifier
        }

        with open(self.data_path+"embedding.json", "w") as f:
            json.dump(save_dict, f)
    
    def load(self, data_path=None, file_name="embedding.json"): # doesnt have safeguards to prevent loading files that dont exist. Expects user to only call this if a valid embedding file is present
        if data_path is None:
            data_path = self.data_path
        with open(data_path+file_name, "r") as f:
            save_dict = json.load(f)
        self.batch_file_paths = save_dict["file_paths"]
        self.batch_idxs = save_dict["batch_idxs"]
        self.identifier = save_dict["identifier"]
            
            