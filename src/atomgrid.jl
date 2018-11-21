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
        R,L
    end
end

export @atomgrid
