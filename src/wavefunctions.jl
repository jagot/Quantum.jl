using LinearAlgebra
using LinearMaps
using ArnoldiMethod
using SphericalOperators
import SphericalOperators: ord, LexicalOrdering

import Base: size, similar, getindex, setindex!, view, reshape, show
import LinearAlgebra: norm, normalize!, ⋅
    
using Printf

struct Wavefunction{T<:Complex,B<:RBasis,S<:AbstractSphericalBasis} <: AbstractVector{T}
    vec::Vector{T}
    basis::B
    L::S
end

Wavefunction(basis::B, L::S) where {B,S} =
    Wavefunction(zeros(complex(eltype(basis)), prod(size(L))), basis, L)

function Wavefunction(ϕ::Eigenstate{T}, ::Type{O}=LexicalOrdering) where {T,O}
    C = complex(T)
    vec = zeros(C,prod(size(ϕ.L)))
    vec[ord(ϕ.L,O,ϕ.ℓ,ϕ.m,1:size(ϕ.L,2))] = ϕ.vec
    Wavefunction(vec, ϕ.basis, ϕ.L)
end

(ψ::Wavefunction)(ℓ::Integer,m::Integer, ::Type{O}=LexicalOrdering) where O =
    view(ψ, ord(ψ.L,LexicalOrdering,ℓ,m,1:size(ψ.L,2)))

size(ψ::Wavefunction, args...) = size(ψ.vec, args...)
similar(ψ::Wavefunction) = Wavefunction(similar(ψ.vec), ψ.basis, ψ.L)

for op in [:getindex, :setindex!, :view, :reshape]
    @eval $op(ψ::Wavefunction, args...) = $op(ψ.vec, args...)
end

reshape(ψ::Wavefunction, ::Colon, i::Int) where N =
    reshape(ψ.vec, :, i)

reshape(ψ::Wavefunction, shape::NTuple{N,<:Int}) where N =
    reshape(ψ.vec, shape)

norm(ψ::Wavefunction) = norm(ψ.vec, ψ.basis)
normalize!(ψ::Wavefunction) = normalize!(ψ.vec)

⋅(a::Wavefunction,b::Wavefunction) = a.vec⋅b.vec
⋅(ϕ::Eigenstate,ψ::Wavefunction) = ϕ.vec⋅ψ(ϕ.ℓ,ϕ.m)
⋅(ψ::Wavefunction,ϕ::Eigenstate) = ψ(ϕ.ℓ,ϕ.m)⋅ϕ.vec

function show(io::IO, ψ::Wavefunction)
    show(io, typeof(ψ))
end

export Wavefunction
