# Bert Distillation

1. `bin` - bash files for running pipelines
2. `configs` - just place configs here
3. `docker` - project Docker files for pure reproducibility
4. `presets` - datasets, notebooks, etc - all you don't need to push to git
5. `requirements` - different project python requirements for docker, tests, CI, etc
6. `scripts` - data preprocessing scripts, utils, everything like `python scripts/.py`
7. `serving` - microservices, etc - production
8. `src` - model, experiment, etc - research

## Usage
```
git clone https://github.com/PUSSYMIPT/bert-distillation.git
cd bert-distillation
pip install -r requirements/requirements.txt
bin/download_lenta.sh
python scripts/split_dataset.py --small
catalyst-dl run -C configs/config_ru_ranger.yml --verbose --distributed
```
It will take a lot of time. "Let's go get some drinks"


## Contribution

1. Clone repository
2. run `pip install -r requirements/requirements-dev.txt -r requirements/requirements.txt`
3. write some code
4. run `catalyst-make-codestyle`
5. run `catalyst-check-codestyle`
6. if exit code is not 0 refactor your code
7. commit!

Also read this at least once
https://www.notion.so/Engineering-Guidelines-cc80b8268eed43d6a96b12aa8444b4ca
