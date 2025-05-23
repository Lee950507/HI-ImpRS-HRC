"""
Delsys EMG Record
=================

This example shows how to record EMG data via Delsys Trigno Control Utility.
"""

import argparse
from datetime import datetime

from record import record

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-a', '--addr',
    dest='host',
    default='192.168.10.10',
    help="IP address of the machine running TCU. Default is localhost.")
args = parser.parse_args()

root_dir = '/data/emg_record'
exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

# For instance, 6 channels, 2000 samples per second and 30 seconds are chosen.
record.record(args.host, 1, 2000, 40, root_dir, exp_name)
