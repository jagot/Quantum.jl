using LinearAlgebra
using LinearMaps
using ArnoldiMethod
using SphericalOperators
import SphericalOperators: ord

import LinearAlgebra: norm, normalize!, getproperty

import Base: isless, show
using Printf

struct Eigenstate{T,R,B<:RBasis,S<:AbstractSphericalBasis,N} <: AbstractVector{T}
    vec::Vector{T}
    E::R
    R::B
    L::S
    quantum_numbers::NTuple{N,Pair{Symbol,<:Integer}}
end
isless(a::Eigenstate, b::Eigenstate) = a.E < b.E

function getproperty(ϕ::Eigenstate, prop::Symbol)
    qns = getfield(ϕ, :quantum_numbers)
    i = findfirst(isequal(prop), first.(qns))
    i === nothing && return getfield(ϕ, prop)
    qns[i][2]
end

function show(io::IO, ϕ::Eigenstate)
    qns = ϕ.quantum_numbers
    write(io, @sprintf("|%s⟩ = |%s⟩, E = %+7.5g au = %+7.5g eV",
                       join(string.(first.(qns)),","),
                       join(string.(last.(qns)),","),
                       ϕ.E, 27.211ϕ.E))
end

norm(v,::FEDVR.Basis) = norm(v, 2)
norm(v,R::BSplines.Basis) = norm(v, 2) # real(dot(v,R.B[end]*v)) # TODO
norm(v,R::FiniteDifferences.Basis) = norm(v, 2)*√(R.ρ)

normalize!(v,::FEDVR.Basis) = normalize!(v, 2)
normalize!(v,R::Union{BSplines.Basis,FiniteDifferences.Basis}) = (v ./= norm(v,R))

include("shift_invert.jl")

function eigenstates(H, R::RBasis, σ::Number=-10.0; kwargs...)
    Hf = construct_linear_map(H, σ)
    ee, = partialschur(Hf; kwargs...)
    λ = 1 ./ diag(ee.R) .+ σ
    for j in 1:size(ee.Q, 2)
        normalize!(@view(ee.Q[:,j]), R)
    end
    λ,ee.Q
end

function ground_state(H, R::RBasis, args...; kwargs...)
    λ,ϕ = eigenstates(H, R, args...; kwargs...)
    real(λ[1]),real.(ϕ[:,1])
end

function ground_state(get_hamiltonian::Function,
                      R::RBasis, L::AbstractSphericalBasis,
                      V::Function,
                      ℓ₀::Integer, m₀::Integer=0,
                      args...; ℓₘₐₓ::Integer=5, kwargs...)
    if ℓ₀ == -1
        candidates = map(0:min(ℓₘₐₓ,L.ℓₘₐₓ)) do ℓ
            ground_state(get_hamiltonian, R, L, V, ℓ, m₀, args...; kwargs...)
        end
        return minimum(candidates)
    end
    ℓ₀ ∉ eachℓ(L) && error("Requested initial partial wave $(ℓ₀) not in range $(eachℓ(L))")
    m₀ ∉ eachm(L,ℓ₀) && error("Requested initial projection quantum number $(m₀) not in range $(eachm(L,ℓ₀))")
    Hℓ₀ = get_hamiltonian(R, ℓ₀, V)
    λ₀,ϕ₀ = ground_state(Hℓ₀, R, args...; kwargs...)
    Eigenstate(ϕ₀, λ₀, R, L, (:n => 1, :ℓ => ℓ₀, :m => m₀))
end

ground_state(R::Union{FEDVR.Basis,FiniteDifferences.Basis}, L::AbstractSphericalBasis,
             V::Function,
             ℓ₀::Integer, m₀::Integer=0,
             args...; kwargs...) =
                 ground_state(hamiltonian, R, L, V, ℓ₀, m₀, args...; kwargs...)

ground_state(R::BSplines.Basis, L::AbstractSphericalBasis,
             V::Function,
             ℓ₀::Integer, m₀::Integer=0,
             args...; kwargs...) =
                 ground_state(R, L, V, ℓ₀, m₀, args...; kwargs...) do R,ℓ₀,V
                     hamiltonian(R,ℓ₀,V),R(I)
                 end

export eigenstates, ground_state, Eigenstate
