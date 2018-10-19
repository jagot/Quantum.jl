using SparseArrays
using SphericalOperators

import SphericalOperators: ord

import FEDVR: basecount, kinop
using BlockMaps
using BandedMatrices

basecount(basis::FEDVR.Basis) = basecount(basis.grid)
basecount(basis::BSplines.Basis) = length(basis.t) - BSplines.order(basis.t)

kinop(basis::BSplines.Basis) = -BSplines.derop(basis, 2)/2

function hamiltonian(basis::RBasis, L::AbstractSphericalBasis,
                     V::Function=coulomb(1.0),
                     ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    nᵣ = basecount(basis)
    @assert nᵣ == size(L,2)
    M = prod(size(L))

    T = kinop(basis) # One-body operator, identical for all partial waves
    if typeof(basis) <: FEDVR.Basis
        # T = BlockMaps.banded(T) # This wastes approximately half the size on zeroes
        T = sparse(T)
    end
    rsel = 1:nᵣ

    # H₀ = BandedMatrix(Zeros(M,M), bandwidths(T))
    H₀ = spzeros(M,M)
    # H₀ = if typeof(basis) <: FEDVR.Basis
    #     spzeros(M,M)
    # else
    #     BandedMatrix(Zeros(M,M), bandwidths(T))
    # end

    for ℓ in eachℓ(L)
        Vℓ = basis(V(ℓ))
        Hℓ = T + Vℓ
        for m in eachm(L,ℓ)
            sel = ord(L,O,ℓ,m,rsel)
            H₀[sel,sel] += Hℓ
        end
    end

    Symmetric(H₀)
end
hamiltonian(basis::RBasis, ℓ::Integer, V::Function=coulomb(1.0); kwargs...) =
    hamiltonian(basis, SphericalBasis2d(ℓ,basecount(basis),ℓₘᵢₙ=ℓ), V; kwargs...)

function interaction_common(fun::Function, basis::RBasis, L::AbstractSphericalBasis, component=:z,
                            ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    m = basecount(basis)
    @assert m == size(L,2)
    M = prod(size(L))
    Hᵢ = spzeros(M,M)

    𝔞,𝔟 = fun()

    op = Dict(:z => SphericalOperators.ζ,
              :x => SphericalOperators.ξ)[component]

    materialize!(Hᵢ, op, L, 𝔞, 𝔟, O)
end


function hamiltonian_E_R(basis::RBasis, L::AbstractSphericalBasis, component=:z,
                         ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    """Dipole interaction Hamiltonian in the length gauge
    Ĥᵢ(t) = 𝓔(t)⋅r, where r = [x,y,z]."""
    interaction_common(basis, L, component, O) do
        R = basis(r -> r)
        rℓ = ℓ -> R
        rℓ,rℓ
    end
end

function APℓ(basis::RBasis, L::AbstractSphericalBasis, component=:z,
             ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    """Dipole interaction Hamiltonian (centrifugal part) in the
    velocity gauge Ĥᵢ(t) = 𝓐(t)⋅p, where p = -im*[∂x,∂y,∂z]."""
    interaction_common(basis, L, component, O) do
        R⁻¹ = basis(r -> 1/r)
        ℓ -> (ℓ+1)*R⁻¹,ℓ -> -ℓ*R⁻¹
    end
end

function ∂ᵣ(basis::RBasis, L::AbstractSphericalBasis, component=:z,
            ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    """Dipole interaction Hamiltonian (differential part) in the
    velocity gauge Ĥᵢ(t) = 𝓐(t)⋅p, where p = -im*[∂x,∂y,∂z]."""
    interaction_common(basis, L, component, O) do
        ∂ᵣop = derop(basis, 1)
        𝔞𝔟 = ℓ -> ∂ᵣop
        𝔞𝔟,𝔞𝔟
    end
end

hamiltonian_A_P(basis::RBasis, L::AbstractSphericalBasis,
                component=:z,
                ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering} =
                    APℓ(basis, L, component, O) + ∂ᵣ(basis, L, component, O)

export hamiltonian, hamiltonian_E_R, hamiltonian_A_P
