__precompile__(true)

module Quantum
using FEDVR

include("orderings.jl")
include("potentials.jl")
include("spherical-tensors.jl")
include("hamiltonians.jl")
include("eigenstates.jl")

end # module
