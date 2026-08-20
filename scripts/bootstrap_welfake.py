"""Download the public WELFake dataset and create local, real classifier assets."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--max-rows', type=int, default=12000)
args = parser.parse_args()

from datasets import load_dataset

root = Path(__file__).resolve().parents[1]
data = load_dataset('davanstrien/WELFake', split='train')
frame = data.to_pandas()[['text', 'label']].dropna()
csv_path = root / 'database' / 'welfake_train.csv'
csv_path.parent.mkdir(exist_ok=True)
frame.to_csv(csv_path, index=False)
subprocess.run([sys.executable, str(root / 'scripts' / 'train.py'), str(csv_path), '--max-rows', str(args.max_rows)], check=True)
