using SparseArrays
using SphericalOperators

import SphericalOperators: ord

function hamiltonian(basis::FEDVR.Basis, L::AbstractSphericalBasis,
                     ::Type{O}=SphericalOperators.LexicalOrdering;
                     v::Function=coulomb(1.0)) where {O<:SphericalOperators.Ordering}
    nᵣ = basecount(basis.grid)
    @assert nᵣ == size(L,2)
    M = prod(size(L))
    H₀ = spzeros(M,M)

    T = sparse(kinop(basis)) # One-body operator, identical for all partial waves
    rsel = 1:nᵣ

    for ℓ in eachℓ(L)
        Vℓ = potop(basis, v(ℓ)).lmap
        Hℓ = T + Vℓ
        for m in eachm(L,ℓ)
            sel = ord(L,O,ℓ,m,rsel)
            H₀[sel,sel] += Hℓ
        end
    end

    H₀
end
hamiltonian(basis::FEDVR.Basis, ℓ::Integer; kwargs...) =
    hamiltonian(basis, SphericalBasis2d(ℓ,basecount(basis.grid),ℓₘᵢₙ=ℓ); kwargs...)

function interaction_common(fun::Function, basis::FEDVR.Basis, L::AbstractSphericalBasis, component=:z,
                            ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    m = basecount(basis.grid)
    @assert m == size(L,2)
    M = prod(size(L))
    Hᵢ = spzeros(M,M)

    𝔞,𝔟 = fun()

    op = Dict(:z => SphericalOperators.ζ,
              :x => SphericalOperators.ξ)[component]

    materialize!(Hᵢ, op, L, 𝔞, 𝔟, O)
end


function hamiltonian_E_R(basis::FEDVR.Basis, L::AbstractSphericalBasis, component=:z,
                         ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    """Dipole interaction Hamiltonian in the length gauge
    Ĥᵢ(t) = 𝓔(t)⋅r, where r = [x,y,z]."""
    interaction_common(basis, L, component, O) do
        R = potop(basis, r -> r).lmap
        rℓ = ℓ -> R
        rℓ,rℓ
    end
end

function APℓ(basis::FEDVR.Basis, L::AbstractSphericalBasis, component=:z,
             ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    """Dipole interaction Hamiltonian (centrifugal part) in the
    velocity gauge Ĥᵢ(t) = 𝓐(t)⋅p, where p = -im*[∂x,∂y,∂z]."""
    interaction_common(basis, L, component, O) do
        R⁻¹ = potop(basis, r -> 1/r).lmap
        ℓ -> (ℓ+1)*R⁻¹,ℓ -> -ℓ*R⁻¹
    end
end

function ∂ᵣ(basis::FEDVR.Basis, L::AbstractSphericalBasis, component=:z,
            ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    """Dipole interaction Hamiltonian (differential part) in the
    velocity gauge Ĥᵢ(t) = 𝓐(t)⋅p, where p = -im*[∂x,∂y,∂z]."""
    interaction_common(basis, L, component, O) do
        ∂ᵣop = sparse(derop(basis, 1))
        𝔞𝔟 = ℓ -> ∂ᵣop
        𝔞𝔟,𝔞𝔟
    end
end

hamiltonian_A_P(basis::FEDVR.Basis, L::AbstractSphericalBasis,
                component=:z,
                ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering} =
                    APℓ(basis, L, component, O) + ∂ᵣ(basis, L, component, O)

export hamiltonian, hamiltonian_E_R, hamiltonian_A_P
