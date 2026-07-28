# /*****************************************************************************/
#  * File: qat_finetune.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-07-23
#  * License: [License Type]
#  * Description: Quantization-aware fine-tuning of a pretrained FalinCNN
#  * checkpoint into FalinCNNQuant (Brevitas, INT8 by default). Reuses the
#  * existing data loading (main.py) and training loop (trainer.py) so QAT
#  * behaves identically to the original float training run, just starting
#  * from the converged float weights instead of from scratch.
#  * See Task 5 of misc/docs/fpga_implementation_plan.md.
#  *
#  * Multi-dataset support (added 2026-07-24): DATASET_NAME is now read from
#  * argv[1] (defaulting to ElectricDevices for backward compat) instead of
#  * being hardcoded, so this same script runs unchanged for any of the 10
#  * UCR datasets used in the paper - just `python qat_finetune.py <Dataset>`.
#  * Requires main.py to have already produced
#  * results/<Dataset>/run_1/FalinCNN_checkpoint.pth (Task 1/2).
# /*****************************************************************************/

import os
import sys
import torch
from torch.utils.data import DataLoader

from main import load_and_encode_ucr, UCRDatasetWrapper
from models.falin_cnn import FalinCNN
from models.falin_cnn_quant import FalinCNNQuant
from trainer import train_standard_model
from utils.config_loader import load_config

CONFIG = load_config()

# ---- Config (dataset comes from argv; hyperparameters from config.json's
# "qat" section - see src/sw/utils/config_loader.py) ----
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else 'ElectricDevices'
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'results', DATASET_NAME, 'run_1', 'FalinCNN_checkpoint.pth')
OUTPUT_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'results', DATASET_NAME, 'run_1', 'FalinCNNQuant_checkpoint.pth')
UCR_ARCHIVE_PATH = os.path.join(BASE_DIR, 'misc', 'data', 'UCRArchive_2018')

QAT_CFG = CONFIG['qat']
WEIGHT_BIT_WIDTH = QAT_CFG['weight_bit_width']
ACT_BIT_WIDTH = QAT_CFG['act_bit_width']

# Fine-tuning from an already-converged float model needs a much lower LR
# and far fewer epochs than training from scratch.
QAT_EPOCHS = QAT_CFG['epochs']
QAT_LR = QAT_CFG['learning_rate']
SCHEDULER_PATIENCE = QAT_CFG['scheduler_patience']
EARLY_STOP_PATIENCE = QAT_CFG['early_stop_patience']
BATCH_SIZE = QAT_CFG['batch_size']


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # --- Load the frozen float checkpoint (dataset/img_strategy/descriptor_size travel with it) ---
    # weights_only=False: PyTorch >=2.6 defaults to True, which rejects the numpy
    # scalars trainer.py's save_model_checkpoint stores (num_classes, label_classes).
    # Safe here since this checkpoint is produced by our own training run.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    img_strategy = checkpoint['img_strategy']
    descriptor_size = checkpoint['descriptor_size']
    num_classes = checkpoint['num_classes']
    label_classes = checkpoint.get('label_classes')

    # NOTE: not trusting checkpoint['dataset'] here - it was saved as "run_1"
    # instead of "ElectricDevices" by a now-fixed trainer.py bug (dataset_name
    # was inferred from os.path.basename(results_dir), which is the run_N
    # folder, not the dataset folder one level up). Checkpoints saved after
    # the trainer.py fix will have the correct value; DATASET_NAME above is
    # authoritative regardless.
    if checkpoint['dataset'] != DATASET_NAME:
        print(f"[!] checkpoint['dataset'] = '{checkpoint['dataset']}' (stale, from the pre-fix trainer.py). "
              f"Using DATASET_NAME = '{DATASET_NAME}' instead.")

    print(f"[*] Loaded float checkpoint: dataset={DATASET_NAME}, img_strategy={img_strategy}, "
          f"descriptor_size={descriptor_size}, num_classes={num_classes}")

    # --- Rebuild the float model and load its trained weights ---
    float_model = FalinCNN(num_classes=num_classes, descriptor_size=descriptor_size)
    float_model.load_state_dict(checkpoint['state_dict'])

    # --- Build the quantized model, warm-started from the float weights ---
    quant_model = FalinCNNQuant.from_pretrained_falin_cnn(
        float_model, weight_bit_width=WEIGHT_BIT_WIDTH, act_bit_width=ACT_BIT_WIDTH
    ).to(device)

    # --- Rebuild the exact same data used for the float run ---
    train_path = os.path.join(UCR_ARCHIVE_PATH, DATASET_NAME, f'{DATASET_NAME}_TRAIN.tsv')
    test_path = os.path.join(UCR_ARCHIVE_PATH, DATASET_NAME, f'{DATASET_NAME}_TEST.tsv')
    X_train, y_train, X_test, y_test, _, _ = load_and_encode_ucr(train_path, test_path)

    train_loader = DataLoader(UCRDatasetWrapper(X_train, y_train, img_strategy),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(UCRDatasetWrapper(X_test, y_test, img_strategy),
                             batch_size=BATCH_SIZE, shuffle=False)

    # --- QAT fine-tuning (reuses the same training loop as the float baseline) ---
    quant_model, best_acc, _ = train_standard_model(
        model=quant_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        model_name="FalinCNNQuant",
        scheduler_patience=SCHEDULER_PATIENCE,
        early_stop_patience=EARLY_STOP_PATIENCE,
        epochs=QAT_EPOCHS,
        lr=QAT_LR,
    )

    print(f"\n[*] QAT fine-tuning complete. Best INT{WEIGHT_BIT_WIDTH} test accuracy: {best_acc * 100:.2f}%")
    print(f"[*] Compare against the float32 baseline in "
          f"results/{DATASET_NAME}/run_1/metrics_summary.txt before moving to export_cnn_weights.py.")

    torch.save({
        'state_dict': quant_model.state_dict(),
        'dataset': DATASET_NAME,
        'img_strategy': img_strategy,
        'descriptor_size': descriptor_size,
        'num_classes': num_classes,
        'label_classes': label_classes,
        'weight_bit_width': WEIGHT_BIT_WIDTH,
        'act_bit_width': ACT_BIT_WIDTH,
    }, OUTPUT_CHECKPOINT_PATH)
    print(f"[*] Quantized checkpoint saved to: {OUTPUT_CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()
