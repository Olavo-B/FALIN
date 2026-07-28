# /*****************************************************************************/
#  * File: config_loader.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-07-24
#  * Description: Single loader for config.json (repo root), the shared source
#  * of truth for software (training/QAT) and hardware (export/RTL)
#  * hyperparameters. Used by main.py and qat_finetune.py directly (both
#  * already live in src/sw/), and by src/hw/cnn/export_cnn_weights.py via its
#  * existing sys.path insert of src/sw/. "_comment" keys in config.json are
#  * documentation only and are not read by this loader.
# /*****************************************************************************/

import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..','..', 'config.json')


def load_config():
    """Returns the parsed config.json as a plain dict."""
    with open(os.path.abspath(_CONFIG_PATH), 'r', encoding='utf-8') as f:
        return json.load(f)
