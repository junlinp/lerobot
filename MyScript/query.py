from datetime import timedelta
from dateutil import parser  # pip install python-dateutil if needed
import wandb
import pandas as pd

api = wandb.Api()
#runs = api.runs("yixuan-he-deepmirror/vast_h100_num_works_test")
#runs = api.runs("yixuan-he-deepmirror/vast_5090_num_works_test")
runs = api.runs("yixuan-he-deepmirror/pick_cup_and_pen")


data = []
for run in runs:
    try:
        created_at = parser.parse(run.created_at)
        heartbeat_at = parser.parse(run.heartbeat_at)
        runtime_sec = (heartbeat_at - created_at).total_seconds()
    except Exception:
        runtime_sec = 0

    if "train/steps" in run.summary:
        train_steps = run.summary.get("train/steps") if  run.summary.get("train/steps") else 1
        sample_per_s = (train_steps * run.config.get("batch_size")) / (runtime_sec)
        data.append({
            "Name": run.name,
            "num_workers": run.config.get("num_workers"),
            "Runtime": str(timedelta(seconds=int(runtime_sec))),
            "train/steps": run.summary.get("train/steps"),
            "batch_size": run.config.get("batch_size"),
            "State": run.state,
            "sample/second": (run.summary.get("train/samples")) / (runtime_sec),
            "gpu": run.config.get("GPU Type", "unknown"),
            "remains": str(timedelta(seconds=(100000 * 64 / sample_per_s -  runtime_sec))),
            "total": str(timedelta(seconds=(100000 * 64 / sample_per_s)))
        })

df = pd.DataFrame(data)
print(df)