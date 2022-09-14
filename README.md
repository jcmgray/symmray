# symmray

A minimal block sparse symmetric tensor python library.

Mostly for learning but other potential motivations:

* use `autoray` both as interface for 'below' (handling the blocks) and 'above'
  (handling the overall `BlockArray` objects)
* efficient fused contractions
* lazy phases
* `.shape` and other `ndarray` looking attributes design
* pythonic naming and design etc.
