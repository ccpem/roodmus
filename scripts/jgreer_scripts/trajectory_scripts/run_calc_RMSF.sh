#!/usr/bin/bash

nohup python calc_RMSF.py --debug=True --chain_id=A --chain_id=B --chain_id=C --chain_id=D --chain_id=E --chain_id=F > run_calc_RMSF.log 2>&1 &