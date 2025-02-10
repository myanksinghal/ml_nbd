import rebound
import pandas as pd
import numpy as np
from tqdm import tqdm

# Function to generate random initial conditions for particles
def generate_random_initial_conditions(num_particles):
    # Generate random positions within [-1,1]
    positions = np.random.rand(num_particles, 3) * 2 - 1  
    # Generate random velocities within [-1,1]
    velocities = np.random.rand(num_particles, 3) * 2 - 1  
    return positions, velocities

# Number of particles and timesteps
num_particles = 3
N_steps = int(1e5)
ejection_threshold = 20.0  # Threshold distance for considering a particle ejected

# Create an initial stable orbit with specified orbital elements
def generate_initial_stable_orbit(m1, m2, a, e, inc, Omega, omega, nu):
    # Ensure orbital parameters are physically valid
    assert e < 1.0
    assert a > 0.0
    assert m1 > 0.0
    assert m2 > 0.0
    assert m1 >= m2

    sim = rebound.Simulation()
    sim.integrator = "ias15"  
    sim.exit_max_distance = ejection_threshold

    # Add primary and secondary bodies
    sim.add(m=m1)
    sim.add(m=m2, a=a, e=e, inc=inc, Omega=Omega, omega=omega, f=nu)

    # Set timestep based on a fraction of the orbit period
    sim.dt = sim.particles[1].P * 1e-2  
    # Restrict minimal timestep
    #sim.ri_ias15.min_dt = 1e-6 * sim.dt
    # Adjust mercury integrator's hill factor
    #sim.ri_mercurius.hillfac = 5

    # Move simulation to center-of-mass frame
    sim.move_to_com()
    return sim

# Function to store simulation data
def sim_store(sim, data):
    total_mass = sum([p.m for p in sim.particles])
    for j, p in enumerate(sim.particles):
        distance = np.sqrt(p.x**2 + p.y**2 + p.z**2)
        velocity = np.sqrt(p.vx**2 + p.vy**2 + p.vz**2)
        # Mark particle as ejected if it crosses the threshold and its velocity is above escape velocity
        ejected = distance > ejection_threshold and velocity > np.sqrt(2 * sim.G * (total_mass-p.m) / distance)

        data.append({
            'simulation': k,
            'timestep': i,
            'particle': j,
            'x': p.x,
            'y': p.y,
            'z': p.z,
            'vx': p.vx,
            'vy': p.vy,
            'vz': p.vz,
            'ejected': ejected
        })
    return data

# List to hold simulation output


# Run one simulation (k=0)
for k in tqdm(np.arange(0, 2000)):
    data = []
    # Generate random angles for orbital elements
    a = np.random.uniform(0.5, 1.5)
    e= np.random.uniform(0, 0.9)
    inc = np.random.uniform(0, np.pi)
    Omega = np.random.uniform(0, 2*np.pi)
    omega = np.random.uniform(0, 2*np.pi)
    nu = np.random.uniform(0, 2*np.pi)

    # Create initial stable orbit
    sim = generate_initial_stable_orbit(1.0, 1.0, 1.0, 0.0, inc, Omega, omega, nu)

    # Add a third body as a perturber
    pert_a=np.random.uniform(5, 10)
    pert_e=np.random.uniform(0.1, 1.5)
    if pert_e > 1.0:
        pert_a=-1*pert_a
    sim.add(m=1.0, a=pert_a, e=pert_e, inc=0.0, Omega=0.0, omega=0.0, f=0.0)
    energy0 = sim.energy()

    with open(f"test_out/simulation_num_{k}.csv", "w") as f:
        f.write("m0,m1,a1,e1,inc1,Omega1,omega1,f1,m2,a2,e2,inc2,Omega2,omega2,f2\n")
        f.write(f"{sim.particles[0].m},{sim.particles[1].m},{sim.particles[1].a}, {sim.particles[1].e}, {sim.particles[1].inc}, {sim.particles[1].Omega}, {sim.particles[1].omega}, {sim.particles[1].f},{sim.particles[2].m}, {sim.particles[2].a}, {sim.particles[2].e}, {sim.particles[2].inc}, {sim.particles[2].Omega}, {sim.particles[2].omega}, {sim.particles[2].f}\n")

        f.close()

    # Integrate for N_steps
    for i in range(N_steps):
        try:
            sim.integrate(sim.t + sim.dt)  # Advance the simulation by one timestep 
            energy = sim.energy()
            delta_energy = np.abs(energy - energy0) / np.abs(energy0)
            if delta_energy > 1e-5:
                print("Energy conservation error")
                with open(f"test_out/simulation_num_{k}.csv", "a") as f:
                    f.write("Energy Error/n")
                    f.close() 
                    break   # Store simulation data
            data = sim_store(sim, data)
        except rebound.Escape as error:
            # If a particle escapes, store data and break
            data = sim_store(sim, data)
            break
        

    # Create a pandas DataFrame
    df = pd.DataFrame(data)


    df.to_csv(f"test_out/simulation_num_{k}.csv",mode='a', index=False)



    # Print the resulting DataFrame
    #print(df)



# Optionally save to CSV

