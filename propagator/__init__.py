import os
from os.path import join, realpath

PROPAGATOR_PATH = realpath(__file__).replace('/propagator/__init__.py', '')
PROPAGATOR_PATH = os.environ.get('PROPAGATOR_PATH', PROPAGATOR_PATH)