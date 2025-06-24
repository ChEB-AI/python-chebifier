# python-chebifier
An AI ensemble model for predicting chemical classes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/python-chebifier.git
cd python-chebifier

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

The package provides a command-line interface (CLI) for making predictions using an ensemble model.

```bash
# Get help
python -m chebifier.cli --help

# Make predictions using a configuration file
python -m chebifier.cli predict example_config.yml --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" "C1=CC=C(C=C1)C(=O)O"

# Make predictions using SMILES from a file
python -m chebifier.cli predict example_config.yml --smiles-file smiles.txt
```

### Configuration File

The CLI requires a YAML configuration file that defines the ensemble model. Here's an example:

```yaml
# Example configuration file for Chebifier ensemble model

# Each key in the top-level dictionary is a model name
model1:
  # Required: type of model (must be one of the keys in MODEL_TYPES)
  type: electra
  # Required: name of the model
  model_name: electra_model1
  # Required: path to the checkpoint file
  ckpt_path: /path/to/checkpoint1.ckpt
  # Required: path to the target labels file
  target_labels_path: /path/to/target_labels1.txt
  # Optional: batch size for predictions (default is likely defined in the model)
  batch_size: 32

model2:
  type: electra
  model_name: electra_model2
  ckpt_path: /path/to/checkpoint2.ckpt
  target_labels_path: /path/to/target_labels2.txt
  batch_size: 64
```

### Python API

You can also use the package programmatically:

```python
from chebifier.ensemble.base_ensemble import BaseEnsemble
import yaml

# Load configuration from YAML file
with open('configs/example_config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Instantiate ensemble model
ensemble = BaseEnsemble(config)

# Make predictions
smiles_list = ["CC(=O)OC1=CC=CC=C1C(=O)O", "C1=CC=C(C=C1)C(=O)O"]
predictions = ensemble.predict_smiles_list(smiles_list)

# Print results
for smile, prediction in zip(smiles_list, predictions):
    print(f"SMILES: {smile}")
    if prediction:
        print(f"Predicted classes: {prediction}")
    else:
        print("No predictions")
```
