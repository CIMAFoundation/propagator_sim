#!/bin/bash
pyarmor pack --name propagator \
        -e "--add-data 'p_vegetation;.' --add-data 'v0_table.txt;.' --add-data 'prob_table.txt;.'" \
        -x "--exclude .venv" \
        main.py 

cp v0_table.txt dist/obf/
cp p_vegetation.txt dist/obf/
cp prob_table.txt dist/obf/