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
from scipy.spatial.transform import Rotation as R

# The following variables can be changed, in order to adjust the simulation.
# --------------------------------------------------------------------------------------------- Solar Cell Properties --

# The solar cell area (in m^2) on the different cubesat faces. Negative areas indicate that the areas point into
# negative directions.
# @formatter:off
solar_cell_area = np.array(
    [    # X            # Y        # Z
        [32e-4,         0,          0       ],     # X+
        [-32e-4,        0,          0       ],     # X-
        [0,             64e-4,      0       ],     # Y+
        [0,             -64e-4,     0       ],     # Y-
        [0,             0,          64e-4   ],     # Z+
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

# The vector that points to the sun as seen from the CubeSat.
SUN_VECTOR = [1.0, 0, 0]

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
ECLIPSE_RATIO = (35 * 60) / ORBIT_PERIOD


# --------------------------------------------------------------------------------------------------------- Functions --

# -------------------------------------------------------------------------------------------------------------- Main --
def main(out_file: str):
    mean_power_per_orbit = np.zeros(COUNT_ROTATIONS, dtype=np.double)
    # Calculate the temperature gradient factor. This is the factor by which the efficiency drops due to temperature
    # degradation.
    temperature_gradient_factor = solar_cell_performance.calc_temperature_gradient_factor(v_op, i_op, v_op_temp_grad,
                                                                                          i_op_temp_grad, t_op,
                                                                                          solar_cell_temp)

    # Generate the time points at which the CubeSat rotation and solar incidence shall be calculated.
    time_points = np.arange(0, ORBIT_PERIOD, STEP_SIZE)

    for i in tqdm(range(COUNT_ROTATIONS)):
        # Generate random rotation rates.
        rot_rate_azi = random.uniform(ROTATION_RATE_MIN, ROTATION_RATE_MAX)
        rot_rate_ele = random.uniform(ROTATION_RATE_MIN, ROTATION_RATE_MAX)
        rot_rate_rol = random.uniform(ROTATION_RATE_MIN, ROTATION_RATE_MAX)

        # Calculate the pose of the CubeSat for each time point.
        pose = [(rot_rate_azi * time_point, rot_rate_ele * time_point, rot_rate_rol * time_point) for time_point in
                time_points]

        # Generate rotation matrices.
        rot_mat = R.from_euler('zyx', pose, degrees=False)

        # Apply rotation to sun vector.
        rotated_sun_vectors = np.array(rot_mat.apply(SUN_VECTOR), dtype=np.double)

        # Project the area vectors onto the sun vector.
        projected_solar_vectors = solar_cell_area @ rotated_sun_vectors.T

        # Negative values in projected_solar_vectors indicate faces that point away from the sun. Set them to 0, as they
        # would not generate any power.
        projected_solar_vectors[projected_solar_vectors < 0.0] = 0.0

        # Sum up the solar area vectors in order to get the cumulative projected area.
        projected_solar_area = projected_solar_vectors.sum(axis=0)

        # Calculate the power values.
        power = projected_solar_area * solar_cell_performance.POWER_DENSITY_AM0 * n * temperature_gradient_factor

        # Calculate the energy per time step.
        energy_per_step = power * STEP_SIZE

        # Calculate the energy per orbit
        energy_per_orbit = energy_per_step.sum() * (1 - ECLIPSE_RATIO)

        # Calculate the mean power per orbit.
        mean_power_per_orbit[i] = (energy_per_orbit / ORBIT_PERIOD)

    m, bins, patches = plt.hist(mean_power_per_orbit, 20)
    plt.xlabel("Mean power per orbit (W)")
    plt.ylabel("Count")
    plt.savefig(out_file, format='svg')
    print("Simulation finished")


# ------------------------------------------------------------------------------------------------------- Entry Point --
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="The output file", default="out.svg")
    args = parser.parse_args()
    main(args.file)
