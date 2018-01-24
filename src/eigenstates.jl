using LinearMaps
using JacobiDavidson
using ILU

function eigenstates(H::LinearMap, pairs::Integer;
                     max_iter=1000, ε=1e-4,
                     solver=bicgstabl_solver,
                     target = SR(zero(Complex128)),
                     real_rotate = true,
                     plot_residuals::Function=r->(),
                     kwargs...)
    if pairs > size(H,2)
        warn("Decreasing pairs from $(pairs) → $(size(H,2))")
        pairs = size(H,2)
    end
    schur, harmonic_ritz_values, converged_ritz_values, residuals = @time jdqr(
        H, solver(size(H,1)),
        pairs = pairs,
        target=target, T=Complex128,
        max_iter=max_iter, ε=ε,
        kwargs...)

    N = min(length(schur.values), size(schur.Q,2))
    N < pairs && warn("Requested $(pairs) Ritz pairs, only $N converged")

    T = real_rotate ? Float64 : Complex128
    f = real_rotate ? real : identity
    λ = zeros(T, N)
    ϕ = zeros(T, (size(H,2), N))

    for i in 1:N
        cᵢ = view(schur.Q, :, i)
        if real_rotate
            # Find the angle of rotation away from the real axis
            φᵢ = atan2(imag(cᵢ[2]),real(cᵢ[2]))
            cᵢ = real.(exp(-im*φᵢ)*cᵢ)
        end
        ϕ[:,i] = normalize(cᵢ)

        λ[i] = f(schur.values[i])
    end

    plot_residuals(residuals)

    λ,ϕ
end


function eigenstates(H::SparseMatrixCSC, pairs::Integer;
                     B=speye(H),
                     max_iter=1000, ε=1e-4,
                     solver=bicgstabl_solver,
                     target = SR(zero(Complex128)),
                     real_rotate = true,
                     plot_residuals::Function=r->(),
                     kwargs...)
    if pairs > size(H,2)
        warn("Decreasing pairs from $(pairs) → $(size(H,2))")
        pairs = size(H,2)
    end
    LU = crout_ilu(H, τ = 0.1)
    schur, residuals = @time jdqz(
        H, B, solver(size(H,1)),
        pairs = pairs,
        target=target,
        preconditioner=LU,
        max_iter=max_iter, ε=ε,
        kwargs...)

    N = min(length(schur.alphas), size(schur.Q.basis,2))
    N < pairs && warn("Requested $(pairs) Ritz pairs, only $N converged")

    T = real_rotate ? Float64 : Complex128
    f = real_rotate ? real : identity
    λ = zeros(T, N)
    ϕ = zeros(T, (size(H,2), N))

    # The left and right eigenvalues/-vectors (alphas,betas and Q,Z,
    # respectively) are the same, since the Hamiltonian is square (?)
    # and B is symmetric (?). (?) = not sure about this statement.

    for i in 1:N
        cᵢ = view(schur.Q.basis, :, i)
        if real_rotate
            # Find the angle of rotation away from the real axis
            φᵢ = atan2(imag(cᵢ[2]),real(cᵢ[2]))
            cᵢ = real.(exp(-im*φᵢ)*cᵢ)
        end
        ϕ[:,i] = normalize(cᵢ)

        λ[i] = f(schur.alphas[i])
    end

    plot_residuals(residuals)

    λ,ϕ
end

function ground_state(H::Union{LinearMap,SparseMatrixCSC}; kwargs...)
    λ,ϕ = eigenstates(H, 1; real_rotate=true, kwargs...)
    real(λ[1]),real.(ϕ[:,1])
end

function ground_state(basis::FEDVR.Basis, ℓs::AbstractVector, ℓ₀::Integer;
                      ordering=lexical_ordering(basis),
                      kwargs...)
    ℓ₀ ∉ ℓs && error("Requested initial partial wave $(ℓ₀) not in range $(ℓs)")
    Hℓ₀ = hamiltonian(basis, ℓ₀)

    m = basecount(basis.grid)
    M = m*length(ℓs)
    ψ₀ = zeros(M)
    ψ₀[ordering.(ℓ₀-ℓs[1],1:m)] = ground_state(Hℓ₀; kwargs...)[2]
    ψ₀
end

export eigenstates, ground_state,
    Near # Reexported from JacobiDavidson.jl for convenience
