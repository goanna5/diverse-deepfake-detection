from pipeline import *
import time

@torch.no_grad()
def evaluate_technique(model, frames_df, technique_function, transform, label_map=None, batch_size=64):
    dataset = Dataset(frames_df, transform, technique_function, label_map)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    correct = predict_correct(model, loader)
    result_df = frames_df.reset_index(drop=True).copy()
    result_df["correct"] = correct
    return result_df

def run_pipeline(models: dict, frames_df, seed=42, label_map=None):
    results = []
    for model_name, (model, transform) in models.items():
        for technique_name, technique_function in AUGMENTATION_TECHNIQUES.items():
            result_df = evaluate_technique(model, frames_df, technique_function, transform, label_map)

            # for all
            results.append({"model": model_name,
                                "technique": technique_name,
                                "gender": "all",
                                "race": "all",
                                "n": len(result_df),
                                "accuracy": result_df["correct"].mean(),
                                })

            # by gender
            for gender, gender_df in result_df.groupby("gender"):
                results.append({"model": model_name,
                                    "technique": technique_name,
                                    "gender": gender,
                                    "race": "all",
                                    "n": len(gender_df),
                                    "accuracy": gender_df["correct"].mean(),
                                    })

            # by race/ethncity
            for race, race_df in result_df.groupby("race"):
                results.append({"model": model_name,
                                    "technique": technique_name,
                                    "gender": "all",
                                    "race": race,
                                    "n": len(race_df),
                                    "accuracy": race_df["correct"].mean(),
                                    })

            # by race/ethnicity and gender
            for (race, gender), race_gender_df in result_df.groupby(["race", "gender"]):
                results.append({"model": model_name,
                                    "technique": technique_name,
                                    "gender": gender,
                                    "race": race,
                                    "n": len(race_gender_df),
                                    "accuracy": race_gender_df["correct"].mean(),
                                    })

     
            print(f"Finished running {model_name}, {technique_name}")

    return pd.DataFrame(results)

def main():
    """ This function should: fetch data, run preprocessing, run pipeline, run results calc
    run combo of each frame * each model * each technique
    """
    label_map = None
    frames_df = get_frames(subset=True, save_imgs=False)
    train_df, test_df = split_by_vid(frames_df, test_portion=0.2)
    print(f"train: {len(train_df)}, test: {len(test_df)}")

    trained_models = {}
    for name, loader_function in MODELS.items():
        start = time.time()
        model = loader_function()
        model = freeze(model, unfreeze_last_n_params=20)
        transform = get_transform_for_timm(model)

        train_dataset = Dataset(train_df, transform, clean, label_map)
        test_dataset = Dataset(test_df, transform, clean, label_map)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        model = fine_tune(model, train_loader, test_loader, epochs=5, model_name=name)
        trained_models[name] = (model, transform)

        finish = time.time()
        print(f"time to run {name}: {finish - start}s")

    results_df = run_pipeline(trained_models, test_df, label_map=label_map)
    filename = f"final_results_{name}_{time.time()}.csv"
    results_df.to_csv(filename, index=False)
    print(results_df)
    


if __name__ == "__main__":
    main()
