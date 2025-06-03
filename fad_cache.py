from pathlib import Path
import fadtk

embedding_model = fadtk.CLAPLaionModel("music")

directories = [
    Path("/data/matt/mg_baseline_output"),
    Path("/data/matt/mg_finetune_output"),
    Path("/data/matt/st_ref"),
]

for d in directories:
    fadtk.cache_embedding_files(
        d,
        ml=embedding_model,
        workers=1,
    )
