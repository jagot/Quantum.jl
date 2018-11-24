using Printf

# * Radial grid
gen_RBasis(::Type{FEDVR.Basis};
           r₀=0.0, rₘₐₓ=0.0, N=0, n=0, kwargs...) =
               FEDVR.Basis(range(r₀,stop=rₘₐₓ,length=N), n)

function gen_RBasis(::Type{FiniteDifferences.Basis};
                    r₀=0.0, rₘₐₓ=0.0, Z=1.0, N=0, ρ=0.0, kwargs...)
    N==0 && ρ==0.0 && throw(ArgumentError("`N` and `ρ` cannot be zero simultaneously"))
    N==0 && (N=ceil(Int, rₘₐₓ/ρ + 1/2))
    ρ==0.0 && (ρ = rₘₐₓ(N - 1/2))
    FiniteDifferences.Basis(N, ρ, Z)
end

gen_RBasis(;grid_type::Symbol=:unknown,kwargs...) =
    gen_RBasis(Dict(
        :fedvr => FEDVR.Basis, :fd => FiniteDifferences.Basis
    )[grid_type]; kwargs...)

# * Angular grid

valid_Lkwargs(::Type{SphericalBasis2d}) = [:m, :ℓₘᵢₙ]
valid_Lkwargs(::Type{SphericalBasis3d}) = [:ℓₘᵢₙ, :mₘₐₓ]

function gen_Lbasis(::Type{S}, ℓₘₐₓ::Integer,
                    R::B, kwargs::Dict{Symbol,<:Any}) where {S<:AbstractSphericalBasis, B<:RBasis}
    ak,vk = keys(kwargs),valid_Lkwargs(S)
    ak ⊆ vk || throw(ArgumentError(
        @sprintf("%s not applicable for %s",
                 join(map(k -> @sprintf("`%s=%s`", k, kwargs[k]),
                          collect(setdiff(ak,vk))), ", "), S)))
    S(ℓₘₐₓ, basecount(R); kwargs...)
end

# * Walkers

atomgrid_walk(node, params, grid_params) = node
function atomgrid_walk(node::Expr, params, grid_params)
    if node.head ∈ [:line, :quote, :macrocall]
        node
    else
        if node.head == :call
            if :grid_type ∉ keys(grid_params)
                grid_params[:grid_type] = node.args[1]
                args = map(node.args[2:end]) do a
                    a isa Expr && a.head == :kw ||
                        error("Unknown grid parameter statement `$(a)`")
                    Expr(:(=), Expr(:ref, grid_params, Expr(:quote, a.args[1])), esc(a.args[2]))
                end
                Expr(:block, args...)
            else
                error("Already specified grid type `$(grid_params[:grid_type])`, cannot also have `$(node.args[1])`")
            end
        elseif node.head == :(=)
            Expr(node.head,
                 Expr(:ref, params, Expr(:quote, node.args[1])),
                 esc(node.args[2]))
        else
            Expr(node.head, atomgrid_walk.(node.args, Ref(params), Ref(grid_params))...)
        end
    end
end

# * Input checkers

function atomgrid_check_params(params::Dict{Symbol,Any})
    :Z ∉ keys(params) && (params[:Z] = 1.0)
    :r₀ ∉ keys(params) && (params[:r₀] = 0)
    for s ∈ [:rₘₐₓ, :ℓₘₐₓ]
        s ∉ keys(params) && error("Must specify `$s`")
    end
end

function atomgrid_check_grid_params(grid_params::Dict{Symbol,Any})
    grid_type = get(grid_params, :grid_type, nothing)
    grid_type === nothing && throw(ArgumentError("Must specify radial grid type"))
    grid_type ∉ [:fedvr,:fd] && throw(ArgumentError("Unknown grid type: `$(grid_type)`"))

    if grid_type == :fedvr
        length([:N,:n] ∩ keys(grid_params)) == 2 ||
            throw(ArgumentError("Must specify `N` and `n` for FEDVR grids"))
    elseif grid_type == :fd
        length([:N,:ρ] ∩ keys(grid_params)) == 1 ||
            throw(ArgumentError("Must specify `N` or `ρ` for FiniteDifferences grids"))
    end
end

# * Grid types
import Base: show, getindex

abstract type AtomGrid{T,B<:RBasis,S<:AbstractSphericalBasis} end

# ** Single gauge
struct SingleGaugeGrid{T,B,S} <: AtomGrid{T,B,S}
    R::B
    L::S
    SingleGaugeGrid(R::B, L::S) where {B,S} = new{eltype(R),B,S}(R,L)
end

function show_extents(io::IO, R::RBasis, L::AbstractSphericalBasis)
    show(io, extents(R))
    write(io, " ⊗ ")
    ℓₘᵢₙ = SphericalOperators.spectroscopic_label(L.ℓₘᵢₙ)
    ℓₘₐₓ = SphericalOperators.spectroscopic_label(L.ℓₘₐₓ)
    write(io, "$(ℓₘᵢₙ)..$(ℓₘₐₓ)")
end

function show(io::IO, grid::SingleGaugeGrid)
    write(io, "Single gauge grid [")
    show_extents(io, grid.R, grid.L)
    write(io, "]\n")
    write(io, "- R: ")
    show(io, grid.R)
    write(io, "\n- L: ")
    show(io, grid.L)
end

# ** Mixed gauge

# This is really ugly, but should go away when bases are implemented
# using ContinuumArrays.jl, when the split can be done just by
# indexing the basis function dimension.

function split_grid(R::FEDVR.Basis, rₛₚₗᵢₜ::T, overlap::T) where T
    g = R.grid
    x = g.X[:,1]
    isplit = findfirst(i -> x[i] ≥ rₛₚₗᵢₜ, elems(g))
    rₛₚₗᵢₜ = x[isplit]
    iap = findlast(i -> x[i] < rₛₚₗᵢₜ-overlap/2, elems(g))
    ier = findfirst(i -> x[i] ≥ rₛₚₗᵢₜ+overlap/2, elems(g))
    (iap === nothing || ier === nothing) &&
        throw(BoundsError("Requested overlap region outside $(extents(R))"))
    overlap = x[ier] - x[iap]
    println((isplit,rₛₚₗᵢₜ))
    println((iap,ier,overlap))
    error("Splitting of FEDVR grids not yet implemented")
    (R, R, R, R, R), rₛₚₗᵢₜ, overlap
end

function split_grid(R::FiniteDifferences.Basis{T}, rₛₚₗᵢₜ::T, overlap::T) where T
    x = locs(R)
    n = length(x)
    isplit = findfirst(i -> x[i] ≥ rₛₚₗᵢₜ, 1:n)
    rₛₚₗᵢₜ = x[isplit]

    ier = findfirst(i -> x[i] ≥ rₛₚₗᵢₜ + overlap/2, 1:n)
    iap = findlast(i -> x[i] < rₛₚₗᵢₜ - overlap/2, 1:n)
    (iap === nothing || ier === nothing) &&
        throw(BoundsError("Requested overlap region outside $(extents(R))"))

    overlap = x[ier]-x[iap]

    Rₗ = FiniteDifferences.Basis(ier, R.ρ, 0, R.Z)
    Rᵥ = FiniteDifferences.Basis(n-iap, R.ρ, iap, R.Z)
    Rₗᵥ = FiniteDifferences.Basis(isplit-iap, R.ρ, iap, R.Z)
    Rᵥₗ = FiniteDifferences.Basis(ier-isplit, R.ρ, isplit, R.Z)
    R̃ₗᵥ = FiniteDifferences.Basis(isplit, R.ρ, 0, R.Z)
    (Rₗ, Rᵥ, Rₗᵥ, Rᵥₗ, R̃ₗᵥ), rₛₚₗᵢₜ, overlap
end

struct MixedGaugeGrid{T,B,S} <: AtomGrid{T,B,S}
    R::B
    L::S
    # Basis for length gauge calculations
    Rₗ::B
    Lₗ::S
    # Basis for velocity gauge calculations
    Rᵥ::B
    Lᵥ::S
    # Basis for transforming from length to velocity gauge
    Rₗᵥ::B
    Lₗᵥ::S
    # Basis for transforming from velocity to length gauge
    Rᵥₗ::B
    Lᵥₗ::S
    # Basis for transforming from length to velocity gauge for observables
    R̃ₗᵥ::B
    L̃ₗᵥ::S

    rₛₚₗᵢₜ::T
    overlap::T
end

# This is ugly, but should go away when product bases are implemented
# using ContinuumArrays.jl.
import SphericalOperators: SphericalBasis2d, SphericalBasis3d
SphericalBasis2d(L::SphericalBasis2d, R::RBasis) =
    SphericalBasis2d(L.ℓₘₐₓ,basecount(R),m=L.m,ℓₘᵢₙ=L.ℓₘᵢₙ)
SphericalBasis3d(L::SphericalBasis3d, R::RBasis) =
    SphericalBasis3d(L.ℓₘₐₓ,basecount(R),ℓₘᵢₙ=L.ℓₘᵢₙ,mₘₐₓ=L.mₘₐₓ)

function MixedGaugeGrid(R::B, L::S, rₛₚₗᵢₜ::Number, overlap::Number) where {B,S}
    T = eltype(R)
    rₛₚₗᵢₜ ∉ extents(R) && throw(BoundsError("Splitting point cannot be outside $(extents(R))"))
    Rs, rₛₚₗᵢₜ, overlap = split_grid(R, T(rₛₚₗᵢₜ), T(overlap))
    Ls = S.(Ref(L), Rs)

    MixedGaugeGrid{T,B,S}(R, L, vcat([[rl...] for rl in zip(Rs, Ls)]...)...,
                          rₛₚₗᵢₜ, overlap)
end

function getindex(grid::MixedGaugeGrid, i::Integer)
    i ∉ 1:2 && throw(ArgumentError("Invalid gauge index $i"))
    i == 1 ? (grid.Rₗ,grid.Lₗ) : (grid.Rᵥ,grid.Lᵥ)
end

getindex(grid::MixedGaugeGrid, ::Colon) = (grid.R,grid.L)

function show(io::IO, grid::MixedGaugeGrid)
    write(io, "Mixed gauge grid [")
    show_extents(io, grid.R, grid.L)
    write(io, "]\n")
    N = 90
    maxR = maximum(locs(grid.R))
    Ner = ceil(Int, N*(grid.rₛₚₗᵢₜ+grid.overlap)/maxR)
    Nsplit = floor(Int, N*grid.rₛₚₗᵢₜ/maxR)
    Nap = floor(Int, N*(grid.rₛₚₗᵢₜ-grid.overlap)/maxR)-1

    write(io, "Er: [",repeat("-",Nsplit),repeat(".",Ner-Nsplit),"]\n")
    write(io, "Ap: ",repeat(" ",Nap-1),"[",repeat(".",Nsplit-Nap+1),repeat("-",N-Nsplit),"]\n")
    write(io, "    ", repeat(" ", Nap-1), "<", repeat("-", Ner-Nap+1), "> ")
    show(io, grid.overlap)
    write(io, " au overlap @ ")
    show(io, grid.rₛₚₗᵢₜ)
    write(io, " au\n")
    for (R,L) in [(:R,:L), (:Rₗ,:Lₗ), (:Rᵥ,:Lᵥ)]
        write(io, @sprintf("\n- %-2s: ", R))
        Robj = getfield(grid, R)
        show(io, Robj)
        write(io, " [")
        show(io, extents(Robj))
        write(io, @sprintf("]\n- %-2s: ", L))
        show(io, getfield(grid, L))
    end
end


# * Frontend macro

macro atomgrid(exprs,dimensions)
    local params = Dict{Symbol,Any}() # General parameters
    local grid_params = Dict{Symbol,Any}() # R-only parameters
    local tree = atomgrid_walk(exprs, params, grid_params)

    quote
        $(tree)()
        atomgrid_check_params($params)
        atomgrid_check_grid_params($grid_params)
        local R = gen_RBasis(;$grid_params..., $params...)
        local Lkwargs = Dict{Symbol,Any}()
        for s ∈ [:m, :ℓₘᵢₙ, :mₘₐₓ]
            s ∈ keys($params) && (Lkwargs[s] = $params[s])
        end
        local L = gen_Lbasis($(esc(dimensions)) == 2 ? SphericalBasis2d : SphericalBasis3d,
                             $(params)[:ℓₘₐₓ], R, Lkwargs)
        if :rₛₚₗᵢₜ ∈ keys($params)
            :overlap ∈ keys($params) || throw(ArgumentError("Must specify `overlap` for mixed gauge grids."))
            MixedGaugeGrid(R, L, $params[:rₛₚₗᵢₜ], $params[:overlap])
        else
            SingleGaugeGrid(R, L)
        end
    end
end

export @atomgrid
