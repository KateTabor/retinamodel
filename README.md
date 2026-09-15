# retinamodel

**A computational framework for testing which retinal circuits best support human-like color judgments across changing scenes.**

## What is retinamodel?

Color can appear remarkably stable even when illumination, shadows, and surrounding surfaces change. `retinamodel` is a research software project for testing where that stability may begin in the visual system.

The project compares biologically grounded retinal circuit models under matched visual conditions. A central question is whether adding S-cone signals to midget-like retinal circuitry improves hue coding and color constancy, and whether different candidate circuits produce distinct patterns of successes and failures.

The broader framework connects retinal circuit models to human and computational color judgments. Visual stimuli are passed through candidate retinal circuits, and the resulting response maps are used by multiple classification models to identify color across changing visual contexts.

> **Project status:** Active development. The retinal front end, synthetic-stimulus generation, human labeling, retinal-response processing, visualization, and automated tests are implemented. The machine-learning and ANN classifiers are being refactored.

## Quick start

The project is currently developed and tested on **macOS**. Other platforms have not yet been tested.

### 1. Get the code and set up the environment

```bash
git clone https://github.com/KateTabor/retinamodel.git
cd retinamodel
uv sync
```

`uv` creates the project environment and installs the required Python packages using the project configuration and locked dependencies.

<details>
<summary><strong>Need Git or uv?</strong></summary>

Install Git from the [official Git website](https://git-scm.com/downloads).

Install `uv` using the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Once both are installed, return to the commands above.

</details>

### 2. Generate the synthetic test dataset

```bash
uv run stimulus --config config/stimulus.toml
```

This generates about 400 synthetic stimuli, runs them through the retinal model, and saves a new dataset under:

```text
data/interm/retina_dataset_<run_id>/
```

The command prints the exact dataset path when it finishes.

Optional check for retinal input calculations:

```bash
uv run mrgc
```

### 3. Choose what to do next

The generated stimulus dataset supports two branches of the workflow.

**Human-in-the-loop color labeling**

```bash
uv run audit start --dataset data/interm/retina_dataset_<run_id>
```

**Retinal-response processing**

```bash
uv run retina-response build --dataset data/interm/retina_dataset_<run_id>
```

**View the retinal responses**

```bash
uv run retina-response view --run data/interm/retina_model_<run_id_2>
```

Replace <run_id> with the dataset run ID printed by stimulus, and <run_id_2> with the retinal-model run ID printed by retina-response build.

> **Storage note:** Retinal-response processing generates large arrays. The command reports the estimated uncompressed storage requirement before processing begins.

## Workflow

The long-term research workflow is:

```text
            Visual stimuli and scenes
                         |
             +-----------+-----------+
             |                       |
             |                       v
             |             Candidate retinal circuits
             |                       |
             |                       v
             v               Retinal response maps
 Human-in-the-loop labeling          |                          
             |                       v
             |             Spatial response processing
             |                       |
             |                       v
             |           Processed retinal responses
             v                       |
             +-----------+-----------+
                         |
                         v
              Classification models
              MLR | SVM | RF | CNN
                         |
                         v
        Compare model predictions with human judgments
                         |
                         v
       Which retinal circuitry best supports
          human-like color judgments?
```

The retinal front end, synthetic-stimulus generation, human labeling, and response-processing stages are implemented. The machine-learning and ANN classifiers are being refactored.

`stim_ID` values link stimuli and their data across stages.

## Commands

| Command | What it does |
| --- | --- |
| `uv run mrgc` | Checks retinal calculations. |
| `uv run stimulus --config config/stimulus.toml` | Generates synthetic stimuli used downstream. |
| `uv run audit --help` | Shows tools for starting, resuming, analyzing, and inspecting human color audits. |
| `uv run retina-response --help` | Shows tools for building and viewing processed retinal-response datasets. |
| `uv run pytest` | Runs the automated test suite. |

Use `--help` with the command-line tools to see their available options.

## Project layout

```text
retinamodel/
├── config/          # Version-controlled model and workflow settings
├── data/
│   ├── raw/         # Source data and permanent human labels
│   ├── interm/      # Generated intermediate datasets and runs
│   └── processed/   # Model outputs and analysis
├── figures/         # Generated figures
├── models/          # Model artifacts and ML/ANN outputs
├── notebooks/       # Research and legacy notebooks
├── src/retinamodel/ # Reusable Python source code
└── tests/           # Automated tests
```

Generated datasets, model outputs, and routine figures are kept out of Git. The code and configuration needed to reproduce them are version controlled.

## Reproducing results and tests

The project is being structured so that the settings used to generate results are explicit and recoverable:

- scientific and processing settings are stored in version-controlled TOML files in `config/`;
- Python dependencies are defined in `pyproject.toml` and locked in `uv.lock`;
- stable `stim_ID` values link data across stages;
- generated runs save metadata and processing settings alongside their outputs; and
- automated tests check the scientific code and command-line workflows.

Run the full test suite with:

```bash
uv run pytest
```

During the refactor, the current synthetic-stimulus and retinal-response stages were also checked directly against outputs from the original research notebook.

## Citation and contact

This project is under active development. Formal citation information will be added with a future research release.

Questions or problems can be shared through [GitHub Issues](https://github.com/KateTabor/retinamodel/issues).

Developed by Kate Tabor in the [Neitz Vision Lab](http://neitzvision.com/) at the University of Washington.
