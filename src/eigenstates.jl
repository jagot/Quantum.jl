using LinearAlgebra
using LinearMaps
using ArnoldiMethod

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

function ground_state(basis::FEDVR.Basis, ℓs::AbstractVector, ℓ₀::Integer, args...;
                      ordering=lexical_ordering(basis),
                      kwargs...)
    ℓ₀ ∉ ℓs && error("Requested initial partial wave $(ℓ₀) not in range $(ℓs)")
    Hℓ₀ = hamiltonian(basis, ℓ₀)

    m = basecount(basis.grid)
    M = m*length(ℓs)
    ψ₀ = zeros(M)
    ψ₀[ordering.(ℓ₀-ℓs[1],1:m)] = ground_state(Hℓ₀, args...; kwargs...)[2]
    ψ₀
end

export eigenstates, ground_state
