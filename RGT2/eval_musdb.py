import musdb
import museval

mus = musdb.DB("/Volumes/FluCoMa 1/datasets/musdb18/")

museval.eval_mus_dir(
    dataset = mus,
    estimates_dir = "sisec_estimates",
    output_dir = "musdb_out",
    subsets = "Test",
    parallel = True
)
