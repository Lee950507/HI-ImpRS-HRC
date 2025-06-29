import os
import time

import numpy as np

import pytrigno
import argparse
from datetime import datetime


def record(host, n, samples_per_read, t, root_path, exp_name):
    """
    Communication with and data acquisition from a Delsys Trigno wireless EMG system.
    Delsys Trigno Control Utility needs to be installed and running on, and the device
    needs to be plugged in. Records can be run with a device connected to a remote machine.

    Args:
        host: host of a remote machine
        n: number of emg channels
        samples_per_read： number of samples per read
        t: number of batches (running time * 2000 / samples_per_read)

    Returns: None
    """
    dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=samples_per_read,
                             host=host)
    dev.set_channel_range((0, n - 1))
    dev.start()
    data_sensor = []
    data_w_time = np.zeros((n + 1, samples_per_read))
    for i in range(int(t)):
        # while True:
        data = dev.read() * 1e6
        system_time = time.time()
        data_w_time[0, :] = system_time
        data_w_time[1:, :] = data
        print(data_w_time)
        assert data_w_time.shape == (n + 1, samples_per_read)
        temp = data_w_time.copy()
        data_sensor.append(temp)        # change to 'data' if system time is not needed
    print(n, '-channel achieved')
    dev.stop()

    data_sensor = np.reshape(np.transpose(np.array(data_sensor), (0, 2, 1)), (-1, n + 1))        # change to 'n' if system time is not needed
    np.save(os.path.join(root_path, exp_name), data_sensor)


if __name__ == '__main__':
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
    record(args.host, 1, 2000, 40, root_dir, exp_name)