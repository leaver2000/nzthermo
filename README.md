# nzthermo

## Getting Started

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install .           
```

### Development

There are some additional tools that useful for development. These can be installed with the
`requirements.dev.txt` file.

```bash
pip install -r requirements.dev.txt
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

Name                     Stmts   Miss  Cover   Missing
------------------------------------------------------
nzthermo/__init__.py         3      0   100%
nzthermo/_c.pyx            112      7    94%   93-95, 196, 203, 222, 234
nzthermo/_typing.py         11      0   100%
nzthermo/const.py           32      0   100%
nzthermo/core.py           192     56    71%   61, 96, 141-142, 270, 283, 300-346, 387-417, 440-441, 458
nzthermo/functional.py     151     39    74%   31, 35, 38-39, 118-119, 132, 144-170, 182-189, 243, 275, 308, 310
------------------------------------------------------
TOTAL                      501    102    80%
```
