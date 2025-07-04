#!/bin/bash

# --- Variables ---
REMOTE_USER="ubuntu"
REMOTE_IP="192.222.52.101"
PEM="~/.ssh/freya.pem"
REMOTE_BASE="~/git"
REMOTE_PROJ="$REMOTE_BASE/balanced-plms"
LOCAL_DATA="/work/lambda/openwebtext/openwebtext-preproc.tgz"
LOCAL_CONFIG="/work/lambda/openwebtext/00_config_openwebtext.json"
LOCAL_MODEL="/work/lambda/ut5-ep15-base.tgz"

# --- 1. Basic SSH Commands ---
ssh -i "$PEM" $REMOTE_USER@$REMOTE_IP <<'EOF'
git config --global user.name "Martin Vejvar"
git config --global user.email "vejvarm@gmail.com"
mkdir -p ~/git
cd ~/git
git clone https://vejvarm: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@github.com/vejvarm/balanced-plms.git || echo "Repo already cloned"
cd balanced-plms
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lambda.txt
wandb login 647dd79b7e09bbee2824b0d06ec9ece9a9cbba66 --relogin
mkdir -p datasets/openwebtext
mkdir -p pretraining/configs
mkdir -p results/t5
EOF

# --- 2. Copy openwebtext-preproc.tgz and config ---
scp -i "$PEM" "$LOCAL_CONFIG" $REMOTE_USER@$REMOTE_IP:$REMOTE_PROJ/pretraining/configs/
scp -i "$PEM" "$LOCAL_MODEL" $REMOTE_USER@$REMOTE_IP:$REMOTE_PROJ/results/t5/
scp -i "$PEM" "$LOCAL_DATA" $REMOTE_USER@$REMOTE_IP:$REMOTE_PROJ/datasets/openwebtext/

# --- 3. Unpack openwebtext-preproc.tgz on remote ---
ssh -i "$PEM" $REMOTE_USER@$REMOTE_IP <<'EOF'
cd ~/git/balanced-plms/results/t5
tar xzvf ut5-ep15-base.tgz
rm -rf ut5-ep15-base.tgz
cd ~/git/balanced-plms/datasets/openwebtext
tar xzvf openwebtext-preproc.tgz
rm -rf openwebtext-preproc.tgz
EOF

echo "Setup of $REMOTE_USER@$REMOTE_IP finished" | mail -s "Server setup complete!" vejvar-martin-km@ynu.jp
