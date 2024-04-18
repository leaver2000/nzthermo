# nzthermo

## Getting Started

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install .           
```

### Development

```bash
python setup.py build_ext --inplace
```

### Testing

```bash
pytest tests
```

### Coverage

In order to compile the cython code for test coverage the code must be compiled with the `--coverage`
flag. This will enable the appropriate compiler flags and macros that allow for code coverage. This
also disables `openmp` which will cause the code to run significantly slower. Unit test can be run
without the `--coverage` flag but the coverage report will not be accurate.

```bash
python setup.py clean --all build_ext --inplace --coverage
coverage run -m pytest
coverage report -m
```
