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
2. run `pip install -r requiriments/requiriments-dev.txt`
3. write some code
4. run `make codestyle`
5. run `make check-codestyle`
6. if exit code is not 0 refactor your code
7. commit!

## Useful links

### Notion
https://www.notion.so/d0a6296fba334043ad696d3844029a2b?v=67f75805cffa46a3b2a78f72d125054f
