# Cholesky decomposition is only valid if A is Hermitian and σ < λₘᵢₙ
get_factorization(A::Union{Symmetric{R,<:BandedMatrix},Hermitian{Complex{R},<:BandedMatrix}}, σ::Number) where {R<:Real} = cholesky(A - I*σ)
get_factorization(A::AbstractMatrix, σ::Number) = factorize(A - I*σ)

struct InverseMap{M} <: LinearMap{M}
    factorization
end
InverseMap(A::AbstractMatrix,σ::Number) = InverseMap{eltype(A)}(get_factorization(A,σ))

Base.size(IM::InverseMap) = size(IM.factorization)

function LinearMaps.A_mul_B!(B, IM::InverseMap, A)
    B[:] = IM.factorization\A
end

# Stolen from https://haampie.github.io/ArnoldiMethod.jl/stable/usage/02_spectral_transformations.html#generalized_shift_invert-1
struct ShiftAndInvert{TA,TB,TT}
    Af::TA
    B::TB
    temp::TT
end

function (M::ShiftAndInvert)(y,x)
    mul!(M.temp, M.B, x)
    # ldiv!(M.Af, M.temp)
    # copyto!(y, M.temp)
    y[:] = M.Af\M.temp
end

function construct_linear_map(A::Union{Symmetric{R,<:AbstractMatrix},Hermitian{Complex{R},<:AbstractMatrix}},B::AbstractMatrix,σ::Number) where {R<:Real}
    Af = get_factorization(A, σ)
    a = ShiftAndInvert(Af,B,Vector{eltype(A)}(undef, size(A,1)))
    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)
end
construct_linear_map(AB::Tuple,σ::Number) = construct_linear_map(AB...,σ::Number)

construct_linear_map(A::Union{AbstractMatrix,LinearMap},σ::Number) = InverseMap(A, σ)
