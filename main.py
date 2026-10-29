"""Fine-tune models and evaluate them across augmentation techniques.
"""

import time

from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import WeightedRandomSampler

from pipeline import *


@torch.no_grad()
def evaluate_technique(model, frames_df, technique_function, transform,
                       label_map=None, batch_size=64):
    """Evaluate a model on frames after applying one augmentation technique.

    Args:
        model: The trained model to evaluate.
        frames_df: DataFrame of frames to evaluate on.
        technique_function: Augmentation function applied to each frame.
        transform: Model-specific preprocessing transform.
        label_map: Optional mapping from label names to class indices.
        batch_size: Number of samples per batch during inference.

    Returns:
        A copy of ``frames_df`` (with a reset index) with extra columns:
        ``pred`` (predicted label), ``label_idx`` (true label index) and
        ``correct`` (whether the prediction matched the true label).
    """
    dataset = Dataset(frames_df, transform, technique_function, label_map)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    preds, labels = predict(model, loader)
    result_df = frames_df.reset_index(drop=True).copy()
    result_df["pred"] = preds
    result_df["label_idx"] = labels
    result_df["correct"] = result_df["pred"] == result_df["label_idx"]
    return result_df


def get_metrics(df):
    """Compute binary precision, recall and F1 for the given results.

    Args:
        df: DataFrame containing ``label_idx`` (true labels) and ``pred``
            (predicted labels) columns. The positive class is ``1``.

    Returns:
        A tuple ``(precision, recall, f1)``.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["label_idx"],
        df["pred"],
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return precision, recall, f1


def make_weighted_sampler(df, label="label"):
    """Build a sampler that balances classes by inverse class frequency.

    Args:
        df: DataFrame containing the label column.
        label: Name of the column holding the label names.

    Returns:
        A ``WeightedRandomSampler`` that samples with replacement, giving
        each sample a weight of ``1 / (count of its class)``.
    """
    labels = df[label].map(LABEL_MAP).values
    class_counts = np.bincount(labels)
    sample_weights = (1.0 / class_counts)[labels]
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def run_pipeline(models: dict, frames_df, seed=42, label_map=None):
    """Evaluate every model on every augmentation technique.

    For each (model, technique) pair, metrics are computed for the whole
    dataset, by gender, by race/ethnicity, and by race/ethnicity and gender.

    Args:
        models: Mapping of model name to a ``(model, transform)`` tuple.
        frames_df: DataFrame of frames to evaluate on. Must contain
            ``gender`` and ``race`` columns.
        seed: Random seed (currently unused).
        label_map: Optional mapping from label names to class indices.

    Returns:
        A DataFrame with one row per (model, technique, gender, race)
        combination and columns for ``n``, accuracy, precision, recall
        and F1. ``"all"`` marks an unfiltered gender or race.
    """
    results = []
    for model_name, (model, transform) in models.items():
        for technique_name, technique_function in (
                AUGMENTATION_TECHNIQUES.items()):
            result_df = evaluate_technique(
                model, frames_df, technique_function, transform, label_map
            )

            # Overall metrics
            precision, recall, f1 = get_metrics(result_df)
            results.append({
                "model": model_name,
                "technique": technique_name,
                "gender": "all",
                "race": "all",
                "n": len(result_df),
                "accuracy": result_df["correct"].mean(),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

            # Metrics by gender
            for gender, gender_df in result_df.groupby("gender"):
                precision, recall, f1 = get_metrics(gender_df)
                results.append({
                    "model": model_name,
                    "technique": technique_name,
                    "gender": gender,
                    "race": "all",
                    "n": len(gender_df),
                    "accuracy": gender_df["correct"].mean(),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })

            # Metrics by race/ethnicity
            for race, race_df in result_df.groupby("race"):
                precision, recall, f1 = get_metrics(race_df)
                results.append({
                    "model": model_name,
                    "technique": technique_name,
                    "gender": "all",
                    "race": race,
                    "n": len(race_df),
                    "accuracy": race_df["correct"].mean(),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })

            # Metrics by race/ethnicity and gender
            for (race, gender), race_gender_df in result_df.groupby(
                    ["race", "gender"]):
                precision, recall, f1 = get_metrics(race_gender_df)
                results.append({
                    "model": model_name,
                    "technique": technique_name,
                    "gender": gender,
                    "race": race,
                    "n": len(race_gender_df),
                    "accuracy": race_gender_df["correct"].mean(),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })

            print(f"Finished running {model_name}, {technique_name}")

    return pd.DataFrame(results)


def main():
    """Run the full experiment: load data, train, evaluate, save results.

    Steps:
        1. Fetch the frames and split them into train/test by video.
        2. Fine-tune each model in ``MODELS`` on the training split.
        3. Evaluate each model on the test split under every augmentation
           technique.
        4. Save the combined results to a timestamped CSV file.
    """
    label_map = None

    # Load data and split by video so frames from one video stay together
    frames_df = get_frames(subset=True, save_imgs=False)
    train_df, test_df = split_by_vid(frames_df, test_portion=0.2)
    print(f"train: {len(train_df)}, test: {len(test_df)}")

    # Fine-tune each model on the clean (un-augmented) training data
    trained_models = {}
    for name, loader_function in MODELS.items():
        start = time.time()
        model = loader_function()
        model = freeze(model, unfreeze_last_n_params=20)
        transform = get_transform_for_timm(model)

        train_dataset = Dataset(train_df, transform, clean, label_map)
        test_dataset = Dataset(test_df, transform, clean, label_map)

        # Weighted sampling balances classes in each training batch
        sampler = make_weighted_sampler(train_df)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, sampler=sampler, num_workers=4
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4
        )

        model = fine_tune(
            model, train_loader, test_loader, epochs=5, model_name=name
        )
        trained_models[name] = (model, transform)

        finish = time.time()
        print(f"time to run {name}: {finish - start}s")

    # Evaluate every model x technique combination on the test split
    results_df = run_pipeline(trained_models, test_df, label_map=label_map)
    filename = f"final_results_{time.time()}.csv"
    results_df.to_csv(filename, index=False)
    print(results_df)


if __name__ == "__main__":
    main()