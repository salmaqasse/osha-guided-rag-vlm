# OSHA-Guided RAG-VLM Construction Safety Screening

A research prototype for image-based screening of five construction-safety categories using Qwen2.5-VL, a curated OSHA rule library, Sentence-BERT embeddings, and FAISS retrieval.

> **Research-use notice:** This project is a screening-support prototype. It is not an official OSHA inspection, citation, legal finding, or substitute for review by a qualified construction-safety professional.

## What the pipeline does

The implementation follows three passes:

1. **Visible-evidence extraction:** Qwen2.5-VL records visible workers, PPE, machinery, edges, trenches, wiring, and spatial cues before making a hazard decision.
2. **OSHA rule retrieval and hazard screening:** Sentence-BERT represents curated rule chunks as embeddings, and FAISS performs similarity search over those chunks. Two prompt-based filters assess the evidence: a strict filter and a cautious filter. Their outputs are reconciled into category-level screening results.
3. **Risk and report generation:** The pipeline produces an advisory severity/probability-based risk result, a concise safety card, and a structured JSON audit record.

The target categories are:

- Fall hazards
- Struck-by / equipment hazards
- Caught-in/between hazards
- Electrical / electrocution hazards
- PPE compliance concern

Housekeeping is outside the current scope.

## Repository contents

```text
OSHA_Guided_RAG_VLM/
├── osha_rag_vlm.py
├── README.md
├── requirements.txt
├── osha_rules.json
├── osha_rules.schema.json
└── .gitignore
```

## OSHA rule library

The repository includes the project rule library, [`osha_rules.json`](osha_rules.json). It contains 27 rule entries across the five target categories and uses the fields documented in [`osha_rules.schema.json`](osha_rules.schema.json).

The Python script's default rules filename is `osha_rules.json`, so the examples below explicitly pass the included file using `--rules-file ./osha_rules.json`. If you use another validated rule library, pass its path with `--rules-file`.

The script checks for these fields at startup:

- `category`
- `official_source`
- `official_source_url`
- `official_text`
- `interpretation`
- `visual_trigger`
- `confirmation_threshold`
- `uncertainty_policy`
- `consistency_rule`
- `source_type`
- `last_verified`

Do not replace the project’s validated rules with invented or unverified rule content. Review the source and usage permissions of the rule text before publishing it.

## Hardware and software requirements

- Python 3.10 or later.
- Linux environment with an NVIDIA GPU and a CUDA-enabled PyTorch installation.
- Enough GPU memory for Qwen2.5-VL-7B-Instruct in 4-bit NF4 mode, including inference overhead. The memory needed depends on image dimensions, processor settings, and the runtime environment.
- Internet access on the first run if the Hugging Face model and processor are not already cached locally and `--allow-download` is used.

**Important:** The current script explicitly stops if CUDA is unavailable. It is not configured to run on CPU-only systems or natively on a Mac without an NVIDIA CUDA environment. GitHub upload can still be performed from a Mac; model inference must run on a compatible CUDA machine.

## Installation

Create and activate a clean environment:

```bash
conda create -n osha-rag-vlm python=3.10 -y
conda activate osha-rag-vlm
```

Install a CUDA-enabled PyTorch build that matches the NVIDIA driver/CUDA environment. Use the official PyTorch installation selector rather than copying a generic command:

- <https://pytorch.org/get-started/locally/>

Then install the remaining Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The code uses FAISS for vector similarity search. If `faiss-cpu` cannot be installed with pip in your environment, use the FAISS installation instructions:

- <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>

## Model

The default model identifier is:

`Qwen/Qwen2.5-VL-7B-Instruct`

The script loads the model in 4-bit NF4 mode using `bitsandbytes`. Model weights are downloaded from Hugging Face only when needed and when `--allow-download` is supplied; otherwise, the script expects the model and processor to be available in the local cache.

Model page: <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>

Follow the model's licence and usage conditions when downloading or using its weights.

## Input images

The script accepts one image, a directory of images, or a text file containing one image path per line. Supported extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

### Run one image

```bash
python osha_rag_vlm.py \
  --image ./example_images/site_image.jpg \
  --allow-download
```

### Run a directory of images

```bash
python osha_rag_vlm.py \
  --image-dir ./example_images \
  --dataset "example_dataset" \
  --split "demo" \
  --allow-download
```

### Run an image list

Create a text file such as `images.txt` with one image path per line. Relative paths are resolved relative to the list file.

```bash
python osha_rag_vlm.py \
  --image-list ./images.txt \
  --output-dir ./output/demo_run \
  --allow-download
```

## Useful command-line options

| Option | Purpose |
|---|---|
| `--image` | Process one image |
| `--image-dir` | Process supported images in a directory |
| `--image-list` | Process image paths listed in a text file |
| `--rules-file` | Set the path to the validated OSHA rules JSON file (default: `osha_rules.json`) |
| `--model-id` | Select a Hugging Face model identifier |
| `--allow-download` | Allow the model/processor to download if not cached |
| `--output-dir` | Choose the output directory (default: `output`) |
| `--output-format` | Choose `json`, `html`, `md`, or `all` (default: `all`) |
| `--ppe-focus` | Mark the run as PPE-focused in metadata; it does not disable other hazard categories |
| `--rules-per-category` | Set the number of retrieved rules per category (1–3; default: 2) |
| `--pass1-max-tokens` | Set the Pass 1 generation token limit (default: 1200) |
| `--verbose` | Print raw intermediate outputs for debugging |

Use `python osha_rag_vlm.py --help` for the complete list.

## Outputs

For each processed image, the script writes a developer/audit JSON file. Depending on `--output-format`, it also writes a concise HTML and/or Markdown safety report. A batch summary JSON is written at the end of the run.

The FAISS index and its key/hash files are generated automatically from the rule library and stored in the working directory. They are rebuildable and should not normally be committed to GitHub.

**Privacy:** JSON audit files can contain local image paths and model outputs. Review and anonymize output files before sharing them. Do not upload private images, datasets, model weights, cached model files, or sensitive audit outputs without the relevant permissions.

## Reproducibility notes

The Python source file was syntax-checked when this repository package was prepared, and the included rule library was checked for valid JSON structure and the required fields. A full end-to-end inference run was not performed here because evaluation images and a compatible CUDA runtime/model environment were not available in this packaging step.

The supplied Python file did not include a package-version lock from the original tested environment. The dependency list is therefore a starting installation list, not a verified reproduction of the original environment. Before publication, record the exact tested versions from the working CUDA environment (for example, with `python -m pip freeze`) and update the dependency file accordingly.

## Citation

If you use this implementation in academic work, cite the associated paper once its publication details are available. The repository URL can then be added to the paper’s Code Availability statement.

## Licence

No software licence has been added because the project owner/co-authors should choose and approve the licence. A public GitHub repository without a licence is viewable, but it does not automatically grant permission to reuse, modify, or redistribute the code.
