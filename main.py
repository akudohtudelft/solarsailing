
import numpy as np
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt
from tqdm import tqdm


DAY = 24*60*60
YEAR = 365*DAY

AU = 1.495979 * 10**11

def rotate(angle, vector):
    return np.matmul(np.array(([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])), vector)

class Body:
    def __init__(self, mu, We, R_ref):
        self.mu = mu
        self.We = We
        self.R_ref = R_ref

    def get_circ_orbital_velocity(self, position):
        r = np.linalg.norm(position)
        v = np.sqrt(self.mu/r)
        r_hat = position/r
        return v*rotate(np.radians(90), r_hat)



class Simulation:
    def __init__(self, area, mass, initial_position: np.ndarray, initial_velocity: np.ndarray, star:Body, inci_angle:float):
        self.position_vector = initial_position
        self.cart_velocity_vector = initial_velocity
        self.star = star
        self.inci_angle = inci_angle
        self.area = area
        self.pointing_vector = self.get_pointing_vector(inci_angle)
        self.mass = mass
        self.pointing_vector = self.get_pointing_vector(self.inci_angle)
        B = self.area * 2 * (self.star.We * self.star.R_ref ** 2) / speed_of_light / self.mass
        self.a_sp = abs(np.cos(self.inci_angle)) * self.pointing_vector * (np.linalg.norm(initial_position) ** (-2)) * B

    def Euler_Rich_step(self, dt):
        an = self.compute_accelerations(self.position_vector, self.cart_velocity_vector)
        vn = self.cart_velocity_vector
        yn = self.position_vector

        v_mid = vn + 0.5 * dt * an
        y_mid = yn + 0.5 * dt * vn

        a_mid = self.compute_accelerations(y_mid, v_mid)

        v_next = vn + dt * a_mid
        y_next = yn + dt * v_mid

        self.cart_velocity_vector = v_next
        self.position_vector = y_next

    def get_pointing_vector(self, incidence_angle):
        r_hat = self.position_vector / np.linalg.norm(self.position_vector)
        pointing_vector = rotate(incidence_angle ,r_hat)
        return pointing_vector


    def compute_accelerations(self, position_vector, cart_velocity_vector):
        r_hat = position_vector / np.linalg.norm(position_vector)
        # gravity lmao :)\

        radial_distance = np.linalg.norm(position_vector)
        a_g = -r_hat * self.star.mu/(radial_distance**2)

        # Solar pressure uwu
        self.pointing_vector = self.get_pointing_vector(self.inci_angle)
        B = self.area * 2 * (self.star.We * self.star.R_ref ** 2) / speed_of_light / self.mass
        self.a_sp = self.pointing_vector*np.cos(self.inci_angle)*(radial_distance**(-2))*B

        # Drag to be implemented...

        return a_g + self.a_sp

    def update_inci_angle(self):
        r = self.position_vector
        v = self.cart_velocity_vector

        r_hat = r / np.linalg.norm(r)
        t_hat = rotate(np.radians(90), r_hat)

        # velocity components
        v_r = np.dot(v, r_hat)
        v_t = np.dot(v, t_hat)

        # incidence angle measured from radial direction
        self.inci_angle = np.arctan2(v_t, v_r)

    def run_for_times(self, start_time, end_time, dt=0.1 * DAY, subdivisions=None):
        if subdivisions is not None:
            dt = (end_time - start_time) / subdivisions

        ts = []
        positions = []
        velocities = []
        accelerations = []

        t = start_time
        n_steps = int((end_time - start_time) / dt)

        for _ in tqdm(range(n_steps), desc="Simulation progress"):
            ts.append(t)
            positions.append(self.position_vector.copy())
            velocities.append(self.cart_velocity_vector.copy())
            accelerations.append(self.a_sp.copy())

            self.Euler_Rich_step(dt)
            t += dt

        self.ts = np.array(ts)
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.accelerations = np.array(accelerations)

    def run_for_times_constant_rp(self, start_time, end_time, dt=0.1 * DAY, subdivisions=None):
        if subdivisions is not None:
            dt = (end_time - start_time) / subdivisions

        ts = []
        positions = []
        velocities = []
        accelerations = []

        t = start_time
        n_steps = int((end_time - start_time) / dt)
        boundary_time = 0.5*YEAR
        for _ in tqdm(range(n_steps), desc="Simulation progress"):
            if t<boundary_time:
                self.inci_angle = -45
            else:
                self.update_inci_angle()
            ts.append(t)
            positions.append(self.position_vector.copy())
            velocities.append(self.cart_velocity_vector.copy())
            accelerations.append(self.a_sp.copy())

            self.Euler_Rich_step(dt)
            t += dt

        self.ts = np.array(ts)
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.accelerations = np.array(accelerations)

    def plot_results(self, t_interval=100):
        positions = self.positions
        velocities = self.velocities
        accelerations = self.accelerations
        ts = self.ts

        # Velocity magnitude for color coding
        v_mag = np.linalg.norm(velocities, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ---- Trajectory plot ----
        sc = ax1.scatter(
            positions[:, 0]/AU,
            positions[:, 1]/AU,
            c=100*v_mag/speed_of_light,
            cmap='viridis',
            s=5
        )

        # Acceleration vectors (sampled)
        idx = np.arange(0, len(ts), t_interval)
        ax1.quiver(
            positions[idx, 0]/AU,
            positions[idx, 1]/AU,
            accelerations[idx, 0],
            accelerations[idx, 1],
            color='red',
            scale=1e-3  # adjust if arrows look too big/small
        )

        ax1.set_xlabel("x position (AU)")
        ax1.set_ylabel("y position (AU)")
        ax1.set_title("Trajectory with Acceleration Vectors")
        ax1.axis('equal')

        cbar = plt.colorbar(sc, ax=ax1)
        cbar.set_label("Velocity magnitude (c%)")

        # ---- Velocity vs time ----
        ax2.plot(ts / YEAR, 100 * v_mag / speed_of_light, label="Velocity (c%)")
        ax2.set_xlabel("Time (Years)")
        ax2.set_ylabel("Velocity magnitude (c%)")
        ax2.set_title("Velocity & Distance over Time")

        # Compute radial distance in AU
        r = np.linalg.norm(positions, axis=1) / AU

        # Secondary y-axis for distance
        ax3 = ax2.twinx()
        ax3.plot(ts / YEAR, r, linestyle='--', label="Distance (AU)")
        ax3.set_ylabel("Distance (AU)")

        # Optional: combined legend
        lines_1, labels_1 = ax2.get_legend_handles_labels()
        lines_2, labels_2 = ax3.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

        plt.tight_layout()
        plt.show()


def main():
    # constants
    S = 100
    We = 1361
    Re = 1.495979 * 10**11
    c = speed_of_light
    m = 1
    B = S*2*(We*Re**2)/c/m
    mu_sun = 1.3271244004210* 10**20

    # differential Equation
    # a = B*r**(-2)

    # Orbital spiraling

    Sun = Body(mu_sun, We, Re)

    initial_position = np.array([1*AU, 0])
    initial_velocity = Sun.get_circ_orbital_velocity(initial_position)

    sim = Simulation(S, m, initial_position, initial_velocity, Sun, np.radians(45))
    sim.run_for_times_constant_rp(0, 100*YEAR)
    sim.plot_results()

def solar_pressure():
    # constants
    S = 10000
    We = 1361
    Re = 1.495979 * 10**11
    c = speed_of_light
    m = 100
    B = S*2*(We*Re**2)/c/m
    mu_sun = 1.3271244004210* 10**20

    # differential Equation
    # a = B*r**(-2)

    # Orbital spiraling

    Sun = Body(mu_sun, We, Re)

    initial_position = np.array([5*AU, 0])
    initial_velocity = Sun.get_circ_orbital_velocity(initial_position)*0.5

    sim = Simulation(S, m, initial_position, initial_velocity, Sun, np.radians(0))
    sim.run_for_times(0, 100*YEAR, dt=0.1*DAY)
    sim.plot_results()

def no_grav_run():
    # constants
    S = 2000000
    We = 1361
    Re = 1.495979 * 10 ** 11
    c = speed_of_light
    m = 0.5
    B = S * 2 * (We * Re ** 2) / c / m
    mu_sun = 1.3271244004210 * 10 ** 20

    # No gravity

    no_grav_Sun = Body(0, We, Re)
    initial_position = np.array([1 * AU, 0])
    initial_velocity = np.array([0, 0])

    sim_no_grav = Simulation(S, m, initial_position, initial_velocity, no_grav_Sun, np.radians(0))
    sim_no_grav.run_for_times(0, 100 * YEAR)
    sim_no_grav.plot_results()


if __name__ == "__main__":
    # main()
    # no_grav_run()
    solar_pressure()
