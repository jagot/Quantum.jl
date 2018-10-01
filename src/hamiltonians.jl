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


function hamiltonian_E_R(basis::FEDVR.Basis, L::AbstractSphericalBasis, component=:z,
                         ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    """Dipole interaction Hamiltonian in the length gauge
    Ĥᵢ(t) = 𝓔(t)⋅r, where r = [x,y,z]."""
    m = basecount(basis.grid)
    @assert m == size(L,2)
    M = prod(size(L))
    Hᵢ = spzeros(M,M)

    R = potop(basis, r -> r).lmap
    rℓ = ℓ -> R

    op = Dict(:z => SphericalOperators.ζ,
              :x => SphericalOperators.ξ)[component]

    materialize!(Hᵢ, op, L, rℓ, rℓ, O)

    Hᵢ
end

function APℓ(basis::FEDVR.Basis, L::AbstractSphericalBasis, component=:z,
             ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    """Dipole interaction Hamiltonian in the velocity gauge
    Ĥᵢ(t) = 𝓐(t)⋅p, where p = -im*[∂x,∂y,∂z]."""
    m = basecount(basis.grid)
    @assert m == size(L,2)
    M = prod(size(L))
    Hᵢ = spzeros(M,M)

    R⁻¹ = potop(basis, r -> 1/r).lmap
    𝔞 = ℓ -> (ℓ+1)*R⁻¹
    𝔟 = ℓ -> -ℓ*R⁻¹

    op = Dict(:z => SphericalOperators.ζ,
              :x => SphericalOperators.ξ)[component]

    materialize!(Hᵢ, op, L, 𝔞, 𝔟)

    Hᵢ
end

function ∂ᵣ(basis::FEDVR.Basis, L::AbstractSphericalBasis, component=:z,
            ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering}
    """Dipole interaction Hamiltonian in the velocity gauge
    Ĥᵢ(t) = 𝓐(t)⋅p, where p = -im*[∂x,∂y,∂z]."""
    m = basecount(basis.grid)
    @assert m == size(L,2)
    M = prod(size(L))
    Hᵢ = spzeros(M,M)

    ∂ᵣop = sparse(derop(basis, 1))
    𝔞𝔟 = ℓ -> ∂ᵣop

    op = Dict(:z => SphericalOperators.ζ,
              :x => SphericalOperators.ξ)[component]

    materialize!(Hᵢ, op, L, 𝔞𝔟, 𝔞𝔟)

    Hᵢ
end

hamiltonian_A_P(basis::FEDVR.Basis, L::AbstractSphericalBasis,
                component=:z,
                ::Type{O}=SphericalOperators.LexicalOrdering) where {O<:SphericalOperators.Ordering} =
                    APℓ(basis, L, component, O) + ∂ᵣ(basis, L, component, O)

export hamiltonian, hamiltonian_E_R, hamiltonian_A_P
