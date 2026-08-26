from pipeline import *
import time

@torch.no_grad()
def evaluate_technique(model, frames_df, technique_function, transform, label_map=None, batch_size=64):
    dataset = Dataset(frames_df, transform, technique_function, label_map)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return evaluate(model, loader)

def run_pipeline(models: dict, frames_df, seed=42, label_map=None):
    
    eval_df = frames_df

    results = []
    for model_name, (model, transform) in models.items():
        for technique_name, technique_function in AUGMENTATION_TECHNIQUES.items():
            acc = evaluate_technique(model, frames_df, technique_function, transform, label_map)

            print(f"{model_name}, {technique_name}, acc={acc}")
            results.append({"model": model_name, "technique": technique_name, "accuracy": acc})

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
