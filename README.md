PROPAGATOR: An Operational Cellular-Automata Based Wildfire Simulator
=====

This repository contains the python implementation of the PROPAGATOR wildfire simulation algorithm, developed by CIMA Research Foundation.

Link to the research paper: [PROPAGATOR: An Operational Cellular-Automata Based Wildfire Simulator](https://www.mdpi.com/2571-6255/3/3/26)

----------------------------------------
## How to install
Clone this repository. Create a virtual environment, activate it and install the required dependendecies with
```bash
pip install -r requirements.txt
```

## Launch a simulation
```bash
python main.py -f ./example/params.json -of ./example/output -tl 24 -dem ./example/dem.tif -veg ./example/veg.tif
```

See `python main.py --help` for command line args.



