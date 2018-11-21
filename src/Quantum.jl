__precompile__(true)

module Quantum
using FEDVR
using BSplines
using FiniteDifferences

RBasis = Union{FEDVR.Basis,BSplines.Basis,FiniteDifferences.Basis}

include("potentials.jl")
include("hamiltonians.jl")
include("eigenstates.jl")
include("wavefunctions.jl")
include("atomgrid.jl")

end # module
