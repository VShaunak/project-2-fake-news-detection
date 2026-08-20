# Model assets

Run `python scripts/train.py path/to/welfake.csv --text-column text --label-column label` to generate the three versioned local model files and measured metrics. The API refuses inference until an asset is available, so a confidence value is never fabricated.

The included lightweight baseline is a reproducible TF-IDF/logistic-regression classifier. Replace it with fine-tuned Transformer checkpoints for a production comparison; record the checkpoint revision and test metrics beside the asset.
