#!/usr/bin/env bash

mkdir -p ./data/vibe_db
export PYTHONPATH="./:$PYTHONPATH"

# AMASS
python lib/data_utils/amass_utils.py --dir /mnt/ExtraDisk/TCMR_data/preprocessed_data/amass


