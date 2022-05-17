git clone https://github.com/fmsnew/nas-bench-nlp-release.git;

mkdir logs_folder
unzip nas-bench-nlp-release/train_logs_multi_runs/logs.zip -d logs_folder;
python3 -m venv nas_repo;
source nas_repo/bin/activate;
pip install -r requirements.txt;

mkdir data;
unzip nas-bench-nlp-release/data/datasets.zip -d data
python nas-rnn/train.py;