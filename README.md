# python-chebifier
An AI ensemble model for predicting chemical classes in the ChEBI ontology. It integrates deep learning models,
rule-based models and generative AI-based models.

A web application for Chebifier is available at https://chebifier.hastingslab.org/.

## Installation

You can get the package from PyPI:
```bash
pip install chebifier[models]
```
If you want the barebones Chebifier without the base learners, run
```bash
pip install chebifier
```
(This is useful if you only need a subset of base learners)

or get the latest development version from GitHub:
```bash
# Clone the repository
git clone https://github.com/yourusername/python-chebifier.git
cd python-chebifier

# Install the package
pip install -e .[models]
```

The Graph Neural Networks depend on `torch_geometric` and `torch_scatter` which you need to install separately ([depending on your CUDA version](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)). E.g.
```bash
pip install torch==2.12.0 torch_scatter torch_geometric -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
```

## Usage

```bash
# Predict for one or more SMILES / InChI strings (default config: web)
python -m chebifier predict -m "CC(=O)OC1=CC=CC=C1C(=O)O" -m "C1=CC=C(C=C1)C(=O)O"

# Predict for molecules listed in a file (one SMILES / InChI per line)
python -m chebifier predict -f smiles.txt

# Use the eval ensemble, or your own configuration file
python -m chebifier predict -e eval -m "CC(=O)O"
python -m chebifier predict -e configs/my_config.yml -f smiles.txt

# Get all available options
python -m chebifier predict --help
```

### Advanced CLI

The ensemble configuration is selected with `--ensemble-config`: `web` or `eval` (both downloaded from
[Hugging Face](https://huggingface.co/datasets/chebai/chebifier), `web` is the default) or a path to your own
configuration file. Create your own file to change which models are included in the ensemble or how they are weighted.

Trained deep learning models are automatically downloaded from [Hugging Face](https://huggingface.co/chebai).
To access a model from Hugging face, add the `load_model` key in your configuration file. For example:

```yaml
my_gat:
  type: gat
  load_model: "gat-aug_chebi25-3star_v252"
```

### Available model weights:

* `gat-aug_chebi25-3star_v252`
* `gat_chebi25-3star_v252`
* `gat-aug_chebi25_v252`
* `gat_chebi25_v252`
* `resgated-aug_chebi25-3star_v252`
* `resgated_chebi25-3star_v252`
* `resgated-aug_chebi25_v252`
* `resgated_chebi25_v252`
* `c3p_with_weights`


You can also supply your own model checkpoints (see `configs/example_config.yml` for an example).

The base learners are selected with `-e`/`--ensemble-config` (default `web`). The deep learning
base learners and the ensemble's calibration for the standard `eval`/`web` configs are downloaded
from Hugging Face automatically on first use. To use a calibration of your own (e.g. one you built
yourself, see below), pass its directory with `-d`/`--ensemble-dir`.

### Python API

You can use the package programmatically as well:

```python
from chebifier.cli import build_base_learners, build_ensemble_model
from chebifier.predict import predict
from chebifier.utils import download_ensemble_calibration

# Base learners from the "web" config ("eval" or a path to your own config also work).
base_learners = build_base_learners("web")
# download_ensemble_calibration() fetches the standard calibration from Hugging Face; pass your own
# directory instead to use a calibration you built yourself.
ensemble = build_ensemble_model("wmv-f1", download_ensemble_calibration(), "web")

smiles_list = ["CC(=O)OC1=CC=CC=C1C(=O)O", "C1=CC=C(C=C1)C(=O)O"]
result = predict(base_learners, ensemble, smiles_list)

# result["predicted_classes"] is the class column space; result["class_decisions"][i] is the
# per-class boolean decision for molecule i.
for i, smiles in enumerate(smiles_list):
    classes = [
        cls
        for cls, keep in zip(result["predicted_classes"], result["class_decisions"][i].tolist())
        if keep
    ]
    print(f"SMILES: {smiles}")
    print(f"Predicted classes: {classes}" if classes else "No predictions")
```

### Ensemble strategies and inconsistency resolution

The strategy that turns the base learner predictions into one ensemble decision is chosen with
`-t`/`--ensemble-type`:

- `mv` — plain majority vote, every model counts equally.
- `wmv-conf` — majority vote weighted by each model's self-reported confidence.
- `wmv-f1` — confidence weighting plus a per-class trust from each model's validation F1 (the default).
- `ltr` — a learning-to-rank meta-model (LambdaMART) fitted on the validation split.
- `des` — dynamic ensemble selection: per molecule, only the locally most competent models vote.

After a decision has been made for each class, the predictions are reconciled with the ChEBI
hierarchy and its disjointness axioms. The method is chosen with `-ir`/`--inconsistency-resolution`
(or disabled with `--no-resolve-inconsistencies`):

- `score-based` — a confidence-based repair of hierarchy and disjointness violations (the default).
- `ilr-godel`, `ilr-lukasiewicz` — iterative local refinement, repairing violations by fuzzy logic.
- `hex` — HEX-graph constrained inference (a bounded approximation).

Both are described in more detail in [The ensemble](#the-ensemble) and
[Inconsistency resolution](#inconsistency-resolution) below.

### Building your own ensemble

To run a new set of models or calibrate on your own data, build an ensemble on the ChEBI validation
split. This writes the calibration (prediction thresholds, class-wise F1 scores, hyperparameters)
into the ensemble directory, which `predict` and `evaluate` then read via `-d`:

```bash
python -m chebifier build -e configs/my_config.yml -t wmv-f1 -d my_ensemble --data-path <dataset>
```

### The models
Currently, the following models are supported:


| Model | Description | #Classes | Publication                                                           | Repository                                                                            |
|-------|-------------|----------|-----------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `electra` | A transformer-based deep learning model trained on ChEBI SMILES strings. | 1,766/2,117*  | [Glauer, Martin, et al., 2024: Chebifier: Automating semantic classification in ChEBI to accelerate data-driven discovery, Digital Discovery 3 (2024) 896-907](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00238a) | [python-chebai](https://github.com/ChEB-AI/python-chebai) |
| `resgated` | A Residual Gated Graph Convolutional Network trained on ChEBI molecules. | 1,766/2,117* | [Khedekar, Aditya Ganesh, 2026: Integrating Chemical Knowledge into Graph Neural Networks, Master Thesis](https://www.uni-osnabrueck.de/fileadmin/informatik/Arbeitsgruppen/Hybride_KI/mt_aditya_khedekar.pdf) | [python-chebai-graph](https://github.com/ChEB-AI/python-chebai-graph) |
| `gat` | A Graph Attention Network trained on ChEBI molecules. | 1,766/2,117* | [Khedekar, Aditya Ganesh, 2026: Integrating Chemical Knowledge into Graph Neural Networks, Master Thesis](https://www.uni-osnabrueck.de/fileadmin/informatik/Arbeitsgruppen/Hybride_KI/mt_aditya_khedekar.pdf) | [python-chebai-graph](https://github.com/ChEB-AI/python-chebai-graph) |
| `chemlog_peptides` | A rule-based model specialised on peptide classes. | 18 | [Flügel, Simon, et al., 2026: Defining Peptides in ChEBI, Jorunal of Cheminformatics](https://link.springer.com/article/10.1186/s13321-026-01196-4) | [chemlog-peptides](https://github.com/sfluegel05/chemlog-peptides) |
| `chemlog_element`, `chemlog_organox` | Extensions of ChemLog for classes that are defined either by the presence of a specific element or by the presence of an organic bond. | 118 + 37 | [Flügel, Simon, et al., 2025: ChemLog: Making MSOL Viable for Ontological Classification and Learning, arXiv](https://arxiv.org/abs/2507.13987) | [chemlog-extra](https://github.com/ChEB-AI/chemlog-extra) |
| `c3p` | A collection _Chemical Classifier Programs_, generated by LLMs based on the natural language definitions of ChEBI classes. | 338 | [Mungall, Christopher J., et al., 2025: Chemical classification program synthesis using generative artificial intelligence, Journal of Cheminformatics](https://link.springer.com/article/10.1186/s13321-025-01092-3) | [c3p](https://github.com/chemkg/c3p) |
| `lopster` | Rules for 36 ChEBI classes, focusing on classes that cannot be expressed in OWL | 36 | [Magka, Despoina, et al., 2014: A rule-based ontological framework for the classification of molecules, Journal of Biomedical Semantics](https://link.springer.com/article/10.1186/2041-1480-5-17) | [original implementation](https://github.com/magkades/lopster) - Chebifier uses an updated version integrated into [chemlog](https://github.com/sfluegel05/chemlog-peptides) |

In addition, Chebifier also includes a ChEBI lookup that automatically retrieves the ChEBI superclasses for a class
matched by a SMILES string. This is not activated by default, but can be included by adding
```yaml
chebi_lookup:
    type: chebi_lookup
    model_weight: 10 # optional
```
to your configuration file.

### The ensemble
The ensemble collects per-class scores from every base learner and turns them into one decision per
class, selected with `-t`/`--ensemble-type`. For an extended description, see
[Flügel, Simon, et al., 2025: Chebifier 2: An Ensemble for Chemistry](https://ceur-ws.org/Vol-4064/SymGenAI4Sci-paper4.pdf).

<img width="700" alt="ensemble_architecture" src="https://github.com/user-attachments/assets/9275d3cd-ac88-466f-a1e9-27d20d67543b" />

| Strategy | How it works |
|----------|--------------|
| `mv` | Plain majority vote; every model that predicted a class counts equally. |
| `wmv-conf` | Majority vote weighted by each model's confidence, i.e. how far its score sits from its calibrated decision threshold (scaled per side so a maximally confident positive and negative both count 1). |
| `wmv-f1` (default) | Confidence weighting plus a per-class trust term, the model's validation F1 raised to the power 6.25. |
| `ltr` | A LambdaMART ranker (adapting [GOLabeler](https://doi.org/10.1093/bioinformatics/bty130)) fitted on the validation split ranks classes per molecule from the base learner scores. Optionally adds per-class validation statistics as features (`class_stats`). |
| `des` | Dynamic ensemble selection (adapting [META-DES.H](https://arxiv.org/pdf/1811.01742)): a meta-classifier estimates each base learner's local competence per molecule, and only the competent ones vote. |

Each model also carries a `model_weight` (configurable, default 1) that scales its vote independently
of the class. `ltr` and `des` calibrate their hyperparameters by 5-fold cross-validation on the
validation split; `chebifier build` takes their constructor arguments as `-ep key=value`. All
strategies emit the same net score, so inconsistency resolution and the decision threshold apply
unchanged.

### Inconsistency resolution
After each class has been decided independently, the predictions are reconciled with the ChEBI
hierarchy (is-a) and disjointness axioms (`data/disjoint_chebi.csv` and `data/disjoint_additional.csv`). The method is chosen with
`-ir`/`--inconsistency-resolution`, or disabled with `--no-resolve-inconsistencies`; each consumes a
net score and returns one, so the decision threshold applies unchanged.

| Method | How it works |
|--------|--------------|
| `score-based` (default) | Repairs hierarchy then disjointness violations by keeping the more confident class of each conflicting pair (confidence = distance from the decision threshold). A final hierarchy pass lowers children rather than raising parents, so no new disjointness conflicts appear. |
| `ilr-godel` | Iterative Local Refinement ([Daniele et al. 2023](https://doi.org/10.1007/s10994-023-06310-3)) with Gödel logic: each subsumption/disjointness constraint is repaired winner-take-all and iterated to a fixpoint. |
| `ilr-lukasiewicz` | The same ILR framework with Łukasiewicz logic, which shares the correction between the two conflicting classes instead of winner-take-all (e.g. scores 0.8/0.7 become 0.55/0.45). |
| `hex` | HEX-graph constrained inference ([Deng et al. 2014](https://doi.org/10.1007/978-3-319-10590-1_4)). Exact inference is intractable on ChEBI's heavily overlapping labels, so this is a bounded branch-and-bound approximation whose intervals decide ties negatively. |
