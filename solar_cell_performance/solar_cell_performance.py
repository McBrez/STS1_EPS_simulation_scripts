########################################################################################################################
#
# @file: solar_cell_performance.py
# @details: Calculates the solar cell power with regards to azimuth angle, elevation angle and temperature.
#
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import argparse

# --------------------------------------------------------------------------------------------------------- Constants --
# The solar power density in space, according to standardized AM0 spectrum. In W/m^2.
POWER_DENSITY_AM0 = 1366.1

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
        [0,             0,          32e-4   ],     # Z+
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

# The step size of the simulation (degree)
step_size = 1


# --------------------------------------------------------------------------------------------------------- Functions --
def get_rotation_matrix_azimuth(angle):
    """
    Rotates the object around the azimuth axis (z-axis) by the given angle.
    :param object: The object that shall be rotated.
    :param angle: The angle in radian.
    :return: A rotated object.
    """
    # @formatter:off
    return np.array(
        [[np.cos(angle),    -1 * np.sin(angle),     0],
         [np.sin(angle),    np.cos(angle),          0],
         [0,                0,                      1]],
        dtype=np.double).T
    # @formatter:on


def get_rotation_matrix_elevation(angle):
    """
    Rotates the object around the elevation axis (y-axis) by the given angle.
    :param object: The object that shall be rotated.
    :param angle: The angle in radian.
    :return: A rotated object.
    """
    return np.array(
        [[np.cos(angle), 0, np.sin(angle)],
         [0, 1, 0],
         [-1 * np.sin(angle), 0, np.cos(angle)]], dtype=np.double).T


# -------------------------------------------------------------------------------------------------------------- Main --
def main(out_file: str):
    # Construct the azimuth/elevation ranges.
    azimuth_degrees = np.arange(start=-179, stop=180, step=step_size, dtype=np.int32)
    elevation_degrees = np.arange(start=-89, stop=90, step=step_size, dtype=np.int32)
    azimuth_radians = azimuth_degrees * np.pi / 180
    elevation_radians = elevation_degrees * np.pi / 180

    # Calculate the temperature gradient factor. This is the factor by which the efficiency drops due to temperature
    # degradation.
    optimum_power = v_op * i_op
    degraded_optimum_power = (v_op + (v_op_temp_grad / 1000.0 * (solar_cell_temp - t_op))) * (i_op + (
            i_op_temp_grad / 1000.0 * (solar_cell_temp - t_op)))
    temperature_gradient_factor = degraded_optimum_power / optimum_power

    # The vector that points to the sun. Unrotated (i.e. elevation and azimuth is zero ) it points into X+ direction.
    sun_vector = np.array([1.0, 0, 0])
    power = np.zeros((len(azimuth_radians), len(elevation_radians)), dtype=np.double)
    for azimuth_idx, azimuth in enumerate(azimuth_radians):
        for elevation_idx, elevation in enumerate(elevation_radians):
            # Rotate the sun vector.
            rotated_sun_vector = get_rotation_matrix_azimuth(azimuth) @ get_rotation_matrix_elevation(
                elevation) @ sun_vector

            # Project the area vector onto the rotated sun vector.
            projected_solar_cell_area = np.dot(solar_cell_area, rotated_sun_vector)

            # Negative values indicate, that the area is completely shaded. Remove those entries.
            projected_solar_cell_area[projected_solar_cell_area < 0.0] = 0.0

            # Sum up the areas.
            solar_cell_area_sum = projected_solar_cell_area.sum()

            # Calculate the power.
            power[
                azimuth_idx, elevation_idx] = solar_cell_area_sum * POWER_DENSITY_AM0 * n * temperature_gradient_factor

    # Calculate min, max and mean
    min = power.min()
    max = power.max()
    mean = power.mean()

    print("Minimum: %f" % min)
    print("Maximum: %f" % max)
    print("Mean: %f" % mean)

    # Construct the plot
    fig, ax = plt.subplots()
    im = ax.imshow(power.T)
    im.set_extent((-179, 179, -89, 89))
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Elevation (°)")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("", rotation=-90, va='bottom')
    plt.xticks(range(-180, 181, 60))
    plt.yticks(range(-90, 91, 30))
    plt.savefig(out_file, format='svg')

    print("Simulation finished")


# ------------------------------------------------------------------------------------------------------- Entry Point --
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="The output file", default="out.svg")
    args = parser.parse_args()
    main(args.file)
