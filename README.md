# TAIGA IACT Likelihood Reconstruction

This repository provides a lightweight likelihood-based reconstruction pipeline for TAIGA IACT shower events.  Cleaned camera images and the accompanying Hillas parameter tables are converted into a template library that can be queried to estimate the shower core location, arrival direction, energy, and depth of maximum.

## Workflow overview

1. **Prepare templates** – list every cleaned event dump (`*_clean_*.txt`) that should contribute to the template library in the JSON configuration.  The loader automatically finds the matching Hillas CSV files.
2. **Train or load the model** – call `run_reconstruction_from_config` with the configuration path.  The function will assemble the template library, fit the likelihood model, and cache it to disk.  Subsequent runs reuse the saved model unless `force_rebuild` is set.
3. **Inspect accuracy** – open the supplied notebook in `notebooks/reconstruction_analysis.ipynb` to load the cached model, reconstruct the test events, and visualise the relative energy error, core offsets, and arrival-direction accuracy.

### Running the reconstruction script

The module can be executed directly to train (or reuse) the cached model and reconstruct the configured test dataset.  By default it looks for `config/reconstruction_config.json` relative to the repository root:

```bash
python3 TAIGA_IACT_likelihood.py
```

To use an alternative configuration, either pass its path as the first argument or set the `TAIGA_LIKELIHOOD_CONFIG` environment variable:

```bash
python3 TAIGA_IACT_likelihood.py path/to/custom_config.json

# or

TAIGA_LIKELIHOOD_CONFIG=path/to/custom_config.json python3 TAIGA_IACT_likelihood.py
```

The command prints where the model and reconstruction table are stored so the notebook can load them afterwards.

### Configuration parameters

The JSON configuration controls both the dataset selection and the optimisation behaviour.  All keys are optional unless noted otherwise:

- `template_txt_files` (**required**): list of paths to the cleaned template dumps.  The corresponding Hillas CSV file is inferred automatically.
- `test_txt_file` (**required**): path to the cleaned dump that should be reconstructed.  Its Hillas CSV file provides the truth labels for evaluation.
- `output_csv` (**required**): where the reconstruction results will be written.
- `model_path` (**required**): location of the cached likelihood model.  The file is created on the first run and reused afterwards.
- `force_rebuild`: set to `true` to ignore any existing cached model and rebuild it from the template files.
- `template_event_limit`: maximum number of template events loaded from the listed files.  Leave this key out (or set it to `null`) to use every available template.  The limit is helpful when quick experiments require reduced runtimes or when memory pressure is a concern.
- `test_event_limit`: processes only the first *N* events from the test file.  This mirrors the template limit to shorten turnaround time when validating changes.
- `random_seed`: seed that controls the stochastic search used by the optimiser so that repeated runs stay reproducible.
- `n_samples`: number of random template-parameter seeds drawn for the initial likelihood exploration.  Larger values improve coverage of the parameter space but increase runtime roughly linearly.
- `max_templates`: cap on the number of template neighbours blended together via inverse-distance weighting.  Restricting the count keeps interpolation stable and avoids diluting the contribution of the most relevant templates.  Set a larger value if the template library is very sparse or highly irregular.
- `refine_iterations`: how many times the optimiser refines the best candidate found so far with local Gaussian proposals.  More iterations can slightly improve fits at the expense of additional likelihood evaluations.
- `refine_radius`: radius (in normalised template space) used for the local refinement proposals.  Smaller radii focus on fine adjustments; larger values enable broader searches around each promising candidate.
- `idw_power`: exponent of the inverse-distance weighting kernel that blends neighbouring template contributions (default: 2).
- `noise_floor`, `pedestal_variance`: camera noise parameters passed to the likelihood model (defaults: 4 and 9 photoelectrons respectively).

### Why do the limits exist?

The likelihood optimiser is stochastic and scales with the number of candidate templates.  When thousands of events are available, evaluating every candidate for each test event can become prohibitively expensive and may even exhaust memory.  The limits (`template_event_limit`, `test_event_limit`, and `max_templates`) make it possible to:

- iterate quickly while developing or tuning the model by constraining the search to a manageable subset;
- prevent extremely distant templates from dominating the interpolation weights and introducing numerical noise;
- keep runtimes predictable when the configuration is reused on larger datasets.

Whenever full precision is required, simply omit the relevant limit or raise it until the runtime/accuracy trade-off becomes acceptable for the given dataset.

### Default hyper-parameters for a 40k-event template library

The configuration bundled with the repository is tuned for a template collection of roughly 40,000 showers.  The defaults strike a balance between coverage and runtime:

- `template_event_limit = null` to ingest every available template entry;
- `n_samples = 2048` so the stochastic search explores a sufficiently dense grid in the six-dimensional parameter space without exhausting runtime budgets;
- `max_templates = 128` to retain a broad yet manageable neighbourhood when interpolating between templates;
- `refine_iterations = 4` with `refine_radius = 0.08` to polish the best candidates without excessive additional evaluations.

Smaller experiments may reduce these numbers to speed up debugging runs, while larger template catalogues can increase them further to sharpen the interpolation.

## Analysis notebook

Open `notebooks/reconstruction_analysis.ipynb` in Jupyter Lab or VS Code to reproduce the diagnostic plots.  The notebook loads the cached likelihood model, reconstructs the configured test events, and derives the accuracy metrics from the generated CSV table.
