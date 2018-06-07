from clawpack.pyclaw import examples
claw = examples.shock_bubble_interaction.setup()
claw.run()
claw.plot()