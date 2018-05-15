import numpy as np
from torch.utils.data import Dataset
import dsdtools
import glob

class DSDDataset(Dataset):
  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.length = len(glob.glob(root_dir+"/*_X.npy"))

  def __len__(self):
     return self.length

  def __getitem__(self, idx):
      base_name = self.root_dir+"/"+str(idx)
      mix = np.load(
        base_name + "_X.npy"
      )
      mix = mix.astype(np.float32)

      mask = np.load(
        base_name + "_Y.npy"
      ).astype(np.long)
      return mix, mask
