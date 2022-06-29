import argparse
import numpy as np
import pandas as pd
import pathlib
import pickle
import torch

LABELS = {
    "cbd": ["0", "1"],
    "cdsc-e": ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"],
    "dyk": ["0", "1"],
    "nkjp": ["placeName", "persName", "orgName", "geogName", "time", "noEntity"],
    "polemo-in": ["__label__meta_zero", "__label__meta_minus_m",
                  "__label__meta_amb", "__label__meta_plus_m"],
    "polemo-out": ["__label__meta_zero", "__label__meta_minus_m",
                   "__label__meta_amb", "__label__meta_plus_m"],
    "psc": ["0", "1"],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_directory",
        type=str,
        required=True,
        help="Directory with pickled test predictions for specified tasks.",
    )
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--task_names", required=True, nargs="+")

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_directory)
    output_dir = pathlib.Path(args.output_directory)

    for task_name in args.task_names:
        logits = pickle.load(open(
            input_dir / f"{task_name}_test_1_predictions.pkl", "rb"
        ))
        logits = torch.tensor(np.array(list(logits.values())))
        # regression tasks
        if len(logits.shape) == 1:
            pred = logits
        # classification tasks
        else:
            probability = torch.nn.functional.softmax(logits)
            labels = torch.argmax(probability, axis=1)
            pred = [LABELS[task_name][l] for l in labels]
        # KLEJ submission requires full names
        if task_name == "nkjp":
            task_name = "nkjp-ner"
        elif task_name == "polemo-in":
            task_name = "polemo2.0-in"
        elif task_name == "polemo-out":
            task_name = "polemo2.0-out"

        pd.DataFrame({'target': pred}).to_csv(
            output_dir / f"test_pred_{task_name}.tsv", index=False
        )
