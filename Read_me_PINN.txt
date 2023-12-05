Welcome to Single Particle Model PINN for fast prototyping!

I am a physics-informed neural network trained to provide the evolution of lithium concentration in active material particles in both electrodes during a discharge process as well as the discharge curve of the full battery, based in the Single Particle Model.

In general, I am trained using NMC811 G-Si chemistry and in a wide range of geometrical parameters and C-rates:
- Negative electrode thickness in [5e-5, 2e-4] m
- Positive electrode thickness in [5e-5, 2e-4] m
- Negative electrode porosity in [0.2, 0.6]
- Positive electrode porosity in [0.2, 0.6]
- C-rate from 1C to 3C

But due to my training process I am capable of providing results even further! Just test me and I will let you know when you are requesting simulations out of my limits.

You can request simulations changing the examples bellow or creating a new cell.
My inputs are:
- Negative electrode thickness (thickness_n) in meters
- Positive electrode thickness (thickness_n) in meters
- Negative electrode porosity (porosity_n)
- Positive electrode porosity (porosity_p)
- C-rate

And my outputs are:
- Lithium concentration in negative electrode ($c_{s,NE}$ in mol/m**3) along the radius in the negative particle ($r_{NE}$ in meters) for an instant of time ($t$ in seconds)
- Lithium concentration in positive electrode ($c_{s,PE}$ in mol/m**3) along the radius in the positive particle ($r_{PE}$ in meters) for an instant of time ($t$ in seconds)
- Battery voltage ($V$ in volts) for an instant of time ($t$ in seconds). 