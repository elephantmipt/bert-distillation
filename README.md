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

1. Clone repository
2. run `pip install -r requirements/requirements-dev.txt`
3. write some code
4. run `catalyst-make-codestyle`
5. run `catalyst-check-codestyle`
6. if exit code is not 0 refactor your code
7. commit!

Also read this at least once
https://www.notion.so/Engineering-Guidelines-cc80b8268eed43d6a96b12aa8444b4ca

## Useful links

### Notion
https://www.notion.so/1d312fd286104ccbb06069602fa83529?v=b2d8d1b70fe34dbda5a6aa401822ebc7
