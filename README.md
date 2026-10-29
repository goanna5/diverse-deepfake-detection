# Deepfake Detection Under Real-World Conditons

My Masters Capstone Project codebase!

#### About the codebase:
- *main.py* - the main file that puts it all together: fetches image frame data, fine-tunes the models, runs the pipeline, computes and saves results.
- *pipeline.py* - the file containing the segments of the pipeline (excluding preprocessing): methods for applying each image degradation, splitting frames by video, loading models, fine-tuning, evaluating, and predicting.
- *pre_processing.py* - the file containing the preprocessing the data from the metadata CSV: adding new fields, removing unnecessary fields, and renaming some fields. Also includes code for extracting and (optionally) saving image frames from the videos.
- *visualise_results.py* - the file for creating my visualisations of the data produced: heatmaps and confusion matrices.
- *meta_data.csv* and *meta_data_subset.csv* - the files containing the dataset metadata, and a subset of the dataset that I made by using the first 2 video ID folders for each combination of gender and ethnicity, and used it for testing that my code actually ran (so I didn't have to run it on the large dataset that FakeAVCeleb is). These aren't actually included in my GitHub repo due to copyright/data use obligations, to access the `meta_data.csv` please request access via https://sites.google.com/view/fakeavcelebdash-lab/dataset-request-usage.

# Referencing
This work couldn't have been completed without the use of the FakeAVCeleb dataset, thank you so much to the creators for providing access and allowing me to use it in my research.

[1]    H. Khalid, S. Tariq, M. Kim, and S. S. Woo, “FakeAVCeleb: A novel audio-video multimodal deepfake dataset,” in *Thirty-fifth Conference on Neural Information 
        Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021. Available: https://openreview.net/forum?id=TAXFsg6ZaOl


# AI Referencing

*main.py* 
- Claude Sonnet 5.5 was used to help with making and implementing (via code scaffold) the weighted random sampler to balance the imbalance in the dataset (16th September 2026).
- Claude Sonnet 5.5 was used to style the code (copy pasted, and asked it to rewrite the exact same code but with good styling, inc. comments and docstrings according to PEP) (29th October 2026).


*pipeline.py*
- Claude Sonnet 4.6 was used to help write the code for compressing, resizing, and adding Gaussian noise (23rd April 2026, 25th May 2026).
- Claude Sonnet 4.6 was also used for the `get_transform_for_timm` function, to ensure all models were consistent for timm (23rd April 2026).
- Claude Sonnet 4.6, Claude Sonnet 5, and GitHub Copilot for auto complete were used in the `freeze`, `fine_tune`, `evaluate`, and `predict` functions to outline code scaffolding that was later modified
somewhat (23rd April 2026-18th August 2026).
- Claude Sonnet 5.5 was used to style the code (copy pasted, and asked it to rewrite the exact same code but with good styling, inc. comments and docstrings according to PEP) (29th October 2026).


*pre_processing.py*
- Claude Sonnet 4.6 was used to help make part of the `extract_img_frames` function (23rd April 2026).
- Claude Sonnet 4.6 was used to help enumerate through the dataset using tqdm for nice loading in `extract_all_img_frames` (23rd April 2026).
- Claude Sonnet 5.5 was used to style the code (copy pasted, and asked it to rewrite the exact same code but with good styling, inc. comments and docstrings according to PEP) (29th October 2026).


*visualise_results.py*
- Claude Sonnet 5.5 was used pretty extensively throughout the whole file to make CSV data into nice looking heatmaps, graphs, and confusion matrices (not my strong suit sorry), although I
did modify some code to avoid unnecessary things I thought werent relevant e.g. making a list to iterate over 2 things to avoid repeated code and making a whole dictionary etc etc in the
process (16th October 2026-29th October 2026).
- Claude Sonnet 5.5 was used to style the code (copy pasted, and asked it to rewrite the exact same code but with good styling, inc. comments and docstrings according to PEP) (29th October 2026).


In all these files, at times I encountered syntax errors, file IO related errors, and errors with packages/libraries not being installed, and Claude Sonnet 5 or 5.5 was used to 
assist in debugging or deciphering error messages.