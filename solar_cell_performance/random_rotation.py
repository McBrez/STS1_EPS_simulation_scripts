########################################################################################################################
#
# @file: random_rotation.py
# @details: Calculates the mean power generation over an orbit, when the cubesat is subjected to random rotation.
#
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import solar_cell_performance
from tqdm import tqdm

# The following variables can be changed, in order to adjust the simulation.
# --------------------------------------------------------------------------------------------- Solar Cell Properties --

# The solar cell area (in m^2) on the different cubesat faces. Negative areas indicate that the areas point into
# negative directions.
# @formatter:off
solar_cell_area = np.array(
    [    # X            # Y        # Z
        [0,         0,          0       ],     # X+
        [-0,        0,          0       ],     # X-
        [0,             0,      0       ],     # Y+
        [0,             0,     0       ],     # Y-
        [0,             0,          64-e-4   ],     # Z+
        [0,             0,          0       ]      # Z-
    ],
    dtype=np.double
)
# @formatter:on

# Solar cell efficiency
n = 0.284
# Solar cell voltage at maximum power point (Volt)
v_op = 2.343
# Solar cell current at maximum power point (Ampere)
i_op = 0.50157
# The reference temperature the maximum power values refer to(°C)
t_op = 28

# Temperature gradient of the optimum power voltage. (mV/°C)
v_op_temp_grad = -6.8
# Temperature gradient of the optimum power current. (mA/°C)
i_op_temp_grad = 0.20

# The temperature of the solar cells. (Celsius)
solar_cell_temp = 60

# --------------------------------------------------------------------------------------------- Simulation Properties --

# The time step size of the simulation (seconds)
STEP_SIZE = 0.1

# The count of different rotations, that shall be simulated.
COUNT_ROTATIONS = 1000

# The maximum rotation rate (radian/s).
ROTATION_RATE_MAX = 2 * np.pi

# The minimum rotation rate (radian/s)
ROTATION_RATE_MIN = -2 * np.pi

# The orbit period (s)
ORBIT_PERIOD = 90 * 60

# The ration of the orbit that is in the shadow of the earth.
ECLIPSE_RATIO = 0.39


# --------------------------------------------------------------------------------------------------------- Functions --

# -------------------------------------------------------------------------------------------------------------- Main --
def main(out_file: str):
    power = np.zeros(COUNT_ROTATIONS, dtype=np.double)
    # Calculate the temperature gradient factor. This is the factor by which the efficiency drops due to temperature
    # degradation.
    temperature_gradient_factor = solar_cell_performance.calc_temperature_gradient_factor(v_op, i_op, v_op_temp_grad,
                                                                                          i_op_temp_grad, t_op,
                                                                                          solar_cell_temp)
    for i in tqdm(range(COUNT_ROTATIONS)):
        # Generate random rotation.
        rot_rate_azi = random.uniform(ROTATION_RATE_MIN, ROTATION_RATE_MAX)
        rot_rate_ele = random.uniform(ROTATION_RATE_MIN, ROTATION_RATE_MAX)
        rot_rate_rol = random.uniform(ROTATION_RATE_MIN, ROTATION_RATE_MAX)

        time_points = np.arange(0, ORBIT_PERIOD, STEP_SIZE)
        energy_per_orbit = 0.0
        for time in time_points:
            # Do rotation.
            projected_solar_area = solar_cell_performance.get_projected_area(solar_cell_area, rot_rate_azi * time,
                                                                             rot_rate_ele * time,
                                                                             rot_rate_rol * time)
            energy_per_orbit += projected_solar_area * solar_cell_performance.POWER_DENSITY_AM0 * n * temperature_gradient_factor * STEP_SIZE

        power[i] = (energy_per_orbit * (1 - ECLIPSE_RATIO)) / ORBIT_PERIOD

    m, bins, patches = plt.hist(power, 20)
    plt.show()
    print("Simulation finished")


# ------------------------------------------------------------------------------------------------------- Entry Point --
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="The output file", default="out.svg")
    args = parser.parse_args()
    main(args.file)
