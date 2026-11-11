# Spec 02 — Training

## 1. Goal
Fine-tune the encoder reproducibly within the compute budget and produce a calibrated classifier.

## 2. Inputs and outputs
Inputs: processed splits, `labels.json`, config sections `model` and `train`.
Outputs under `artifacts/model/`: model weights, tokenizer, `labels.json`, `calibration.json` (`{"temperature": float}` plus validation ECE and NLL before and after scaling), `training_meta.json` (git SHA, data version, config, run ID, device, `model_version`).

`model_version` is `<base model name>-<data version>-<first 8 characters of the training run ID>`.

The resumable checkpoint is `artifacts/checkpoints/last.pt` (model, optimiser, scheduler, scaler, step, best score). It is reused only when `train.resume` is true and the `model` and `train` config sections, the data version and the training-set size are unchanged; it is deleted when training finishes. The batch order of each epoch depends only on `train.seed` and the epoch number, so a resumed run sees the same batches.

## 3. Procedure
1. Set seeds for Python, NumPy and PyTorch from `train.seed`.
2. Tokenise with truncation at `model.max_length`.
3. Train with AdamW, linear warm-up (`train.warmup_ratio`) then linear decay, mixed precision when `train.device` is `cuda`.
   When `train.micro_batch_size` is set, each batch is processed in chunks of that size and gradients are accumulated before the single optimiser step. Batch size, step count, batch order and schedule are unchanged; it only lowers GPU memory.
4. Evaluate on validation every `train.eval_every_steps` and once more at the end if the last step was not evaluated; keep the checkpoint with the best macro F1.
5. Stop early after `train.early_stopping_patience` evaluations without improvement.
6. Save a resumable checkpoint every `train.save_every_steps` (needed for Colab and Kaggle).
7. `train.device: auto` selects CUDA, then MPS, then CPU.

## 4. Logging
Per `specs/04-experiment-tracking.md`: loss and learning rate per `train.log_every_steps`; validation accuracy and macro F1 at each evaluation; total seconds; peak GPU memory.

## 5. Calibration
After training, collect validation logits from the best checkpoint and fit one temperature by minimising negative log-likelihood. Log calibration error before and after. Test data is not used.

## 6. Hyperparameter search
Grid over `train.lr` in {1e-5, 2e-5, 5e-5} and `train.epochs` in {3, 5, 8}, one seed each, selected on validation macro F1. Final configuration is rerun with three seeds.

## 7. Acceptance
- NFR-2 compute budget met on the reference GPU.
- A resumed run reaches the same best validation score as an uninterrupted run, within noise.
- Unit test trains a tiny random model for a few steps on CPU and checks that the output files exist.
