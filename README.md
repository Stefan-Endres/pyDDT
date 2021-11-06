pyddt
====
Python scripts for deflagration-to-detonation transition simulations.
![Figure_1](https://user-images.githubusercontent.com/8235072/140623846-a912833b-7498-44a0-a451-f6789edea37b.png)


Instructions
====

Simulation of deflagration-to-detonation transition DDT systems using the model developed Sáenz and Steward (2008) with a 5th order WENO scheme for the spatial discretisation. After cloning the scripts to a local directory the system can be run using:

```
$ python ddt.py
```

The current parameter set is for the PETN system. A new system can by added by specifying all the parameters and initial conditions in the `params.py` file. Additionally, a reference energy `e_0` for the equation of state must be recomputed for every system (including changing initial conditions) by running `$ python wreos.py` and addeding the output of the final integral to the `e_0` value in `params.py`.

Validations
===
Additional vadilation for `WRKR` integrator object in `integrator.py` was provided by simulation the 1D test cases `tddt#.py` which replicated the results from Peng et al. (2009). These cases additionally demontrate how the integrator can be adapted to new 1D test problems.

References
====

- Sáenz, J. A., & Stewart, D. S. (2008). Modeling deflagration-to-detonation transition in granular explosive pentaerythritol tetranitrate. Journal of Applied Physics, 104(4), 043519.
- Peng, J., Zhai, C., Ni, G., Yong, H., & Shen, Y. (2019). An adaptive characteristic-wise reconstruction weno-z scheme for gas dynamic euler equations. Computers & Fluids, 179, 34-51.
