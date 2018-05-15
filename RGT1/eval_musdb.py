import musdb
import museval
from settings import *

mus = musdb.DB(dataset_path)

museval.eval_mus_dir(
    dataset = mus,
    estimates_dir = "sisec_estimates", 
    output_dir = "musdb_out",
    subsets = "Test",
    parallel = True
)
