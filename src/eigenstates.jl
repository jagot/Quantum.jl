using LinearAlgebra
using LinearMaps
using ArnoldiMethod
using SphericalOperators
import SphericalOperators: ord

struct InverseMap{M} <: LinearMap{M}
    factorization
end
InverseMap(A::AbstractMatrix) = InverseMap{eltype(A)}(factorize(A))

Base.size(IM::InverseMap) = size(IM.factorization)

function LinearMaps.A_mul_B!(B, IM::InverseMap, A)
    B[:] = IM.factorization\A
end

function eigenstates(H, σ=-10.0; kwargs...)
    Hs = H - sparse(I,size(H)...)*σ
    Hf = InverseMap(Hs)

    ee, = partialschur(Hf; kwargs...)

    λ = 1 ./ diag(ee.R) .+ σ

    λ,ee.Q
end

function ground_state(H, args...; kwargs...)
    λ,ϕ = eigenstates(H, args...; kwargs...)
    real(λ[1]),real.(ϕ[:,1])
end

function ground_state(basis::FEDVR.Basis, L::AbstractSphericalBasis,
                      ℓ₀::Integer, m₀::Integer=0,
                      ::Type{O}=SphericalOperators.LexicalOrdering,
                      args...; kwargs...) where {O<:SphericalOperators.Ordering}
    ℓ₀ ∉ eachℓ(L) && error("Requested initial partial wave $(ℓ₀) not in range $(eachℓ(L))")
    m₀ ∉ eachm(L,ℓ₀) && error("Requested initial projection quantum number $(m₀) not in range $(eachm(L,ℓ₀))")
    Hℓ₀ = hamiltonian(basis, ℓ₀)

    nᵣ = basecount(basis.grid)
    M = prod(size(L))
    ψ₀ = zeros(M)
    ψ₀[ord(L,O,ℓ₀,m₀,1:nᵣ)] = ground_state(Hℓ₀, args...; kwargs...)[2]
    ψ₀
end

export eigenstates, ground_state
