git clone https://github.com/fmsnew/nas-bench-nlp-release.git;

mkdir logs_folder;
unzip nas-bench-nlp-release/train_logs_multi_runs/logs.zip -d logs_folder;

mkdir data;
unzip nas-bench-nlp-release/data/datasets.zip -d data;

mkdir logs;

python3 -m venv nas_repo;
source nas_repo/bin/activate;
pip install -r nas-rnn/requirements.txt;

python nas-rnn/train.py;