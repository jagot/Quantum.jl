using LinearAlgebra
using LinearMaps
using ArnoldiMethod
using SphericalOperators
import SphericalOperators: ord

import LinearAlgebra: norm, normalize!

norm(v,::FEDVR.Basis) = norm(v, 2)
norm(v,basis::BSplines.Basis) = real(dot(v,basis.B[end]*v))

normalize!(v,::FEDVR.Basis) = normalize!(v, 2)
normalize!(v,basis::BSplines.Basis) = (v /= norm(v,basis))

include("shift_invert.jl")

function eigenstates(H, σ::Number=-10.0; kwargs...)
    Hf = construct_linear_map(H, σ)
    ee, = partialschur(Hf; kwargs...)
    λ = 1 ./ diag(ee.R) .+ σ
    λ,ee.Q
end

function ground_state(H, args...; kwargs...)
    λ,ϕ = eigenstates(H, args...; kwargs...)
    real(λ[1]),real.(ϕ[:,1])
end

function ground_state(get_hamiltonian::Function,
                      basis::RBasis, L::AbstractSphericalBasis,
                      V::Function,
                      ℓ₀::Integer, m₀::Integer=0,
                      ::Type{O}=SphericalOperators.LexicalOrdering,
                      args...; kwargs...) where {O<:SphericalOperators.Ordering}
    ℓ₀ ∉ eachℓ(L) && error("Requested initial partial wave $(ℓ₀) not in range $(eachℓ(L))")
    m₀ ∉ eachm(L,ℓ₀) && error("Requested initial projection quantum number $(m₀) not in range $(eachm(L,ℓ₀))")
    Hℓ₀ = get_hamiltonian(basis, ℓ₀, V)

    nᵣ = basecount(basis)
    M = prod(size(L))
    ψ₀ = zeros(M)
    ψ₀[ord(L,O,ℓ₀,m₀,1:nᵣ)] = ground_state(Hℓ₀, args...; kwargs...)[2]
    normalize!(ψ₀, basis)
end

ground_state(basis::FEDVR.Basis, L::AbstractSphericalBasis,
             V::Function,
             ℓ₀::Integer, m₀::Integer=0,
             ::Type{O}=SphericalOperators.LexicalOrdering,
             args...; kwargs...) where {O<:SphericalOperators.Ordering} =
                 ground_state(hamiltonian, basis, L, V, ℓ₀, m₀, O, args...; kwargs...)

ground_state(basis::BSplines.Basis, L::AbstractSphericalBasis,
             V::Function,
             ℓ₀::Integer, m₀::Integer=0,
             ::Type{O}=SphericalOperators.LexicalOrdering,
             args...; kwargs...) where {O<:SphericalOperators.Ordering} =
                 ground_state(basis, L, V, ℓ₀, m₀, O, args...; kwargs...) do basis,ℓ₀,V
                     hamiltonian(basis,ℓ₀,V),basis(I)
                 end

export eigenstates, ground_state
