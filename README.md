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

### Usage

```python
import numpy as np
import nzthermo as nzt
pressure = np.array(
    [1013, 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 600, 550, 500, 450, 400, 350, 300],
) # (Z,)
pressure *= 100
temperature = np.array( 
    [
        [243, 242, 241, 240, 239, 237, 236, 235, 233, 232, 231, 229, 228, 226, 235, 236, 234, 231, 226, 221, 217, 211],
        [250, 249, 248, 247, 246, 244, 243, 242, 240, 239, 238, 236, 235, 233, 240, 239, 236, 232, 227, 223, 217, 211],
        [293, 292, 290, 288, 287, 285, 284, 282, 281, 279, 279, 280, 279, 278, 275, 270, 268, 264, 260, 254, 246, 237],
        [300, 299, 297, 295, 293, 291, 292, 291, 291, 289, 288, 286, 285, 285, 281, 278, 273, 268, 264, 258, 251, 242],
    ]
) # (N, Z)
dewpoint = np.array(
    [
        [224, 224, 224, 224, 224, 223, 223, 223, 223, 222, 222, 222, 221, 221, 233, 233, 231, 228, 223, 218, 213, 207],
        [233, 233, 232, 232, 232, 232, 231, 231, 231, 231, 230, 230, 230, 229, 237, 236, 233, 229, 223, 219, 213, 207],
        [288, 288, 287, 286, 281, 280, 279, 277, 276, 275, 270, 258, 244, 247, 243, 254, 262, 248, 229, 232, 229, 224],
        [294, 294, 293, 292, 291, 289, 285, 282, 280, 280, 281, 281, 278, 274, 273, 269, 259, 246, 240, 241, 226, 219],
    ]
) # (N, Z)
nzt.downdraft_cape(pressure, temperature, dewpoint) #(N,)
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
