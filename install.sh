conda env create -f environment.yml -p ./.conda
conda activate ./.conda 
pip install --upgrade pip setuptools wheel packaging
pip install -r requirements.txt
