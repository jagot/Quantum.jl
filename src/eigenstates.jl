using LinearAlgebra
using LinearMaps
using ArnoldiMethod
using SphericalOperators
import SphericalOperators: ord

import LinearAlgebra: norm, normalize!

norm(v,::FEDVR.Basis) = norm(v, 2)
norm(v,basis::BSplines.Basis) = norm(v, 2) # real(dot(v,basis.B[end]*v)) # TODO
norm(v,basis::FiniteDifferences.Basis) = norm(v, 2)*√(basis.ρ)

normalize!(v,::FEDVR.Basis) = normalize!(v, 2)
normalize!(v,basis::Union{BSplines.Basis,FiniteDifferences.Basis}) = (v ./= norm(v,basis))

include("shift_invert.jl")

function eigenstates(H, basis::RBasis, σ::Number=-10.0; kwargs...)
    Hf = construct_linear_map(H, σ)
    ee, = partialschur(Hf; kwargs...)
    λ = 1 ./ diag(ee.R) .+ σ
    for j in 1:size(ee.Q, 2)
        normalize!(@view(ee.Q[:,j]), basis)
    end
    λ,ee.Q
end

function ground_state(H, basis::RBasis, args...; kwargs...)
    λ,ϕ = eigenstates(H, basis, args...; kwargs...)
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
    ψ₀[ord(L,O,ℓ₀,m₀,1:nᵣ)] = ground_state(Hℓ₀, basis, args...; kwargs...)[2]
    ψ₀
end

ground_state(basis::Union{FEDVR.Basis,FiniteDifferences.Basis}, L::AbstractSphericalBasis,
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
