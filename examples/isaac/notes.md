


Argument Parser:
- Load argument parser from isaaclab app.AppLauncher
- This provides good standard baseline for passing arguments, always use this
- Always good to set height and width as new arguments, lower resolution is good
  for high performance, high resolution if needed

Main scene:
1. Call scene configuration
2. Call function that places primitives
3. Call reset once
4. Step through while the simulation is running

Each object gets its own configuration, where it can be used to define its
properties, shaders etc.

One can define a single configuration and apply to any number of USD files, use
this to spawn objects without having to overload too many objects at once.


Objects:
- Rigid objects: Static objects
- Deformable objects: Object with motion
- Translation, orientation: tuples that describe its location on the map

Default objects located in /World/ - not sure of naming convention.

Distant light as a source source for now.

Apply size tuple, height and width for object sizing. Note sure if its one or
the other, or they are separate definitions.
