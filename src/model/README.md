# Prediction Final Model Production 

This repository contains the production training script and configuration for the finalized paddy yield prediction model.

---

## Project Structure

* **`config.yaml`**: Configuration file defining data paths, target variables, grouping keys, and model parameters.
* **`final_model.py`**: Production training script that fits the final model on the full dataset and serializes artifacts.

---

## How to Run

```bash
uv run final_model.py
```