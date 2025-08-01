import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')


# -------------------------------
# 1. Domain & Simulation Setup
# -------------------------------
def setup_domain():
    domain_x, domain_y, domain_z = 250e3, 250e3, 50e3  # meters
    num_x, num_y, num_z = 100, 100, 20
    delta_x, delta_y, delta_z = domain_x / num_x, domain_y / num_y, domain_z / num_z

    # Physical wind (m/s)
    vel_x, vel_y, vel_z = 20.0, 10.0, 0.0

    # Compute dt from CFL condition
    cfl_x = delta_x / abs(vel_x) if vel_x != 0 else np.inf
    cfl_y = delta_y / abs(vel_y) if vel_y != 0 else np.inf
    cfl_z = delta_z / abs(vel_z) if vel_z != 0 else np.inf
    delta_t = 0.5 * min(cfl_x, cfl_y, cfl_z)
    print(f"Δt = {delta_t:.2f} s per step")

    return num_x, num_y, num_z, delta_x, delta_y, delta_z, vel_x, vel_y, vel_z, delta_t


# -------------------------------
# 2. Temperature Initialization
# -------------------------------
def init_temperature(mode, num_x, num_y, num_z, delta_z):
    if mode == "blob":
        temp_field = np.zeros((num_x, num_y, num_z))
        center_x, center_y, center_z = num_x // 2, num_y // 2, num_z // 2
        radius_x, radius_y, radius_z = 10, 10, 4

        for i in range(num_x):
            for j in range(num_y):
                for k in range(num_z):
                    if ((i - center_x) ** 2 / radius_x ** 2 +
                        (j - center_y) ** 2 / radius_y ** 2 +
                        (k - center_z) ** 2 / radius_z ** 2) <= 1.0:
                        temp_field[i, j, k] = 1.0
        return temp_field

    if mode == "realistic":
        base_temp = 300.0  # K at bottom
        perturb_amp = 15.0  # ±K amplitude of variations
        top_cool = 30.0  # K drop from bottom to top (~50 km)

        # 3D smoothed random field for horizontal patterns
        noise = np.random.rand(num_x, num_y, num_z)
        noise: np.ndarray = gaussian_filter(noise, sigma=(2, 2, 1))  # finer patterns
        noise = (noise - noise.mean()) / np.ptp(noise)
        noise *= perturb_amp

        # Base + horizontal variations
        temp_field = base_temp + noise

        # Gentle vertical gradient
        for k in range(num_z):
            temp_field[:, :, k] -= top_cool * (k / (num_z - 1))
        return temp_field

    return None


# -------------------------------
# 3. Visualization Helpers
# -------------------------------
def plot_initial_slice(temp_field, z_index):
    plt.figure(figsize=(6, 6))
    plt.imshow(temp_field[:, :, z_index].T, origin='lower', cmap='coolwarm')
    plt.colorbar(label='Temperature [K]')
    plt.title(f"Initial temperature at Z={z_index}")
    plt.show()


def animate_vertical_slices(temp_field):
    num_x, num_y, num_z = temp_field.shape
    frame_list = [temp_field[:, :, k] for k in range(num_z)]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frame_list[0].T, origin='lower', cmap='coolwarm',
                   vmin=temp_field.min(), vmax=temp_field.max())
    plt.colorbar(im, ax=ax, label='Temperature [K]')
    text = ax.text(0.02, 0.95, '', color='white', fontsize=12, transform=ax.transAxes)

    def update(frame_idx):
        im.set_data(frame_list[frame_idx].T)
        text.set_text(f"Z-layer {frame_idx + 1}/{num_z}")
        return [im, text]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_list), interval=500, blit=True)
    plt.show()
    return ani  # Keep reference alive


# -------------------------------
# 4. Advection Simulation
# -------------------------------
def advect_temperature(temp_field, num_x, num_y, num_z, delta_x, delta_y, delta_z,
                       vel_x, vel_y, vel_z, delta_t, num_steps, z_index):
    frame_list = [temp_field[:, :, z_index].copy()]  # include initial frame

    for _ in tqdm(range(num_steps)):
        new_temp_field = temp_field.copy()
        for i in range(1, num_x - 1):
            for j in range(1, num_y - 1):
                for k in range(1, num_z - 1):
                    u_plus, u_minus = max(vel_x, 0), min(vel_x, 0)
                    flux_x = (u_plus * (temp_field[i, j, k] - temp_field[i - 1, j, k]) +
                              u_minus * (temp_field[i + 1, j, k] - temp_field[i, j, k])) / delta_x

                    v_plus, v_minus = max(vel_y, 0), min(vel_y, 0)
                    flux_y = (v_plus * (temp_field[i, j, k] - temp_field[i, j - 1, k]) +
                              v_minus * (temp_field[i, j + 1, k] - temp_field[i, j, k])) / delta_y

                    w_plus, w_minus = max(vel_z, 0), min(vel_z, 0)
                    flux_z = (w_plus * (temp_field[i, j, k] - temp_field[i, j, k - 1]) +
                              w_minus * (temp_field[i, j, k + 1] - temp_field[i, j, k])) / delta_z

                    new_temp_field[i, j, k] = temp_field[i, j, k] - delta_t * (flux_x + flux_y + flux_z)

        # Periodic boundary conditions
        new_temp_field[0, :, :] = new_temp_field[-2, :, :]
        new_temp_field[-1, :, :] = new_temp_field[1, :, :]
        new_temp_field[:, 0, :] = new_temp_field[:, -2, :]
        new_temp_field[:, -1, :] = new_temp_field[:, 1, :]
        new_temp_field[:, :, 0] = new_temp_field[:, :, -2]
        new_temp_field[:, :, -1] = new_temp_field[:, :, 1]

        temp_field = new_temp_field
        frame_list.append(temp_field[:, :, z_index].copy())
    return frame_list


# -------------------------------
# 5. Animation of Time Evolution
# -------------------------------
def animate_time_evolution(frame_list, z_index, delta_t, anomaly_mode=True):
    if anomaly_mode:
        frame_list = [frame - frame.mean() for frame in frame_list]
        cmap = 'RdBu_r'
    else:
        cmap = 'hot'

    data_stack = np.stack(frame_list)
    mask = data_stack != 0
    vmin, vmax = data_stack[mask].min(), data_stack[mask].max()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frame_list[0].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    def update(frame_idx):
        im.set_data(frame_list[frame_idx].T)
        sim_time_min = frame_idx * delta_t / 60.0
        ax.set_title(f"Z-slice={z_index} | Time={sim_time_min:.1f} min")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_list), interval=100, blit=False)
    plt.show()
    return ani  # Keep reference alive


# -------------------------------
# Main Execution
# -------------------------------
def main():
    num_x, num_y, num_z, delta_x, delta_y, delta_z, vel_x, vel_y, vel_z, delta_t = setup_domain()
    mode = "realistic"  # "blob" or "realistic"
    temp_field = init_temperature(mode, num_x, num_y, num_z, delta_z)

    z_index = num_z // 2
    # plot_initial_slice(temp_field, z_index)
    # animate_vertical_slices(temp_field)  # optional

    num_steps = 100
    frame_list = advect_temperature(temp_field, num_x, num_y, num_z,
                                    delta_x, delta_y, delta_z,
                                    vel_x, vel_y, vel_z, delta_t, num_steps, z_index)
    _ = animate_time_evolution(frame_list, z_index, delta_t, anomaly_mode=True)


if __name__ == "__main__":
    main()
