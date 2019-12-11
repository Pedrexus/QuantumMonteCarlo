# Quantum Monte Carlo

Here you will find a script for the exercise 2.3 of the book Monte Carlo Methods in Ab Initio Quantum
Chemistry.

## Installation

1. (*Recommended*) Create a virtual environment 

```bash
virtualenv venv
```

or 

```bash
virtualenv --python=python3 venv
```
 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install -r requirements.txt
```

## Usage

At the *scripts* folder, first run ```metropolis.py``` and plot the result with ```plotting.py```.

At ```metropolis.py```, change the parameters at lines *275 - 277* to achieve better results.
```python
n_steps = 10000 
ensemble_size = 1000
delta = 1e-5
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)