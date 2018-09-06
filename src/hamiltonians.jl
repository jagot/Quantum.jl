using SparseArrays

function hamiltonian(basis::FEDVR.Basis, ℓs::AbstractVector;
                     v::Function=coulomb(1.0),
                     ordering=lexical_ordering(basis))
    T = kinop(basis) # One body operator, identical for all partial waves

    m = basecount(basis.grid)
    M = m*length(ℓs)
    H₀ = spzeros(M,M)

    ℓ₀ = ℓs[1]
    for ℓ in ℓs
        Vℓ = potop(basis, v(ℓ)).lmap
        for b in T.blocks
            nn = size(b.a,2)
            H₀[ordering.(ℓ-ℓ₀,(1:nn) .+ (b.i-1)), ordering.(ℓ-ℓ₀,(1:nn) .+ (b.j-1))] += b.a
        end
        H₀[ordering.(ℓ-ℓ₀,1:m),ordering.(ℓ-ℓ₀,1:m)] += Vℓ
    end

    H₀
end
hamiltonian(basis::FEDVR.Basis, ℓ::Integer; kwargs...) = hamiltonian(basis, [ℓ]; kwargs...)

function hamiltonian_E_R(basis::FEDVR.Basis, ℓs::AbstractVector;
                         ordering=lexical_ordering(basis))
    """Dipole interaction Hamiltonian in the length gauge
    Ĥᵢ(t) = 𝓔(t)⋅r, where r = [x,y,z]."""
    R = potop(basis, r -> r)

    m = basecount(basis.grid)
    M = m*length(ℓs)
    Hᵢ = spzeros(M,M)

    ℓ₀ = ℓs[1]
    for ℓ in ℓs[1:end-1]
        zℓ = C(1,ℓ,ℓ+1)*R.lmap
        Hᵢ[ordering.(ℓ-ℓ₀,1:m),ordering.(ℓ-ℓ₀+1,1:m)] = zℓ
        Hᵢ[ordering.(ℓ-ℓ₀+1,1:m),ordering.(ℓ-ℓ₀,1:m)] = zℓ
    end

    Hᵢ
end

export hamiltonian, hamiltonian_E_R
