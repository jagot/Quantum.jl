__precompile__(true)

module Quantum
using FEDVR
using BSplines

RBasis = Union{FEDVR.Basis,BSplines.Basis}

include("potentials.jl")
include("hamiltonians.jl")
include("eigenstates.jl")

end # module
