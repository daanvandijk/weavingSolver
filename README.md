# weavingSolver
Aims to solve the following weaving problem:

## Problem description
_"
  Given is an input image, a number of available wires and specified dimensions of the desired product.
  The weaving machine can only use a maximum number of wires -- let's say 5 -- in its weaving pattern.
  The aim is to recreate the input image with the weaving machine with minimal distortion.
  We can assume that a 'pixel' in the fabric gets build up from 20% blocks of wires. 
  E.g., a 'fabric pixel' may contain 20% color from wire 1, 60% color from wire 3, and 20% color from wire 5.
"_

*Remark:*
Notice that the colors of the wires can be represented in [0,1]-cubed, which is a Hilbert space.
Clearly, if the number of available wires is the same as the maximum wires that can be used by the machine, this is a trivial problem.
This problem becomes interesting when the number of available wires is much greater than the maximal wires that _can_ be used by the machine.
