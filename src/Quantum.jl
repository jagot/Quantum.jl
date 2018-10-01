__precompile__(true)

module Quantum
using FEDVR

include("potentials.jl")
include("hamiltonians.jl")
include("eigenstates.jl")

end # module
