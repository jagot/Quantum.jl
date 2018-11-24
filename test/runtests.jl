using Quantum
import Quantum: extents
using IntervalSets
using Statistics
using Test

@testset "Mixed gauge grid" begin
    function test_mixed_gauge(RL,split,overlap,σ)
        println(RL)
        @test extents(RL[1][1]) ∪ extents(RL[2][1]) == extents(RL[:][1])
        ov = extents(RL[1][1]) ∩ extents(RL[2][1])

        @test mean(ov) ≥ split
        @test isapprox(mean(ov), split, atol=σ)

        lov = abs(-(endpoints(ov)...))
        @test lov ≥ overlap
        @test isapprox(lov, overlap, atol=σ)

        @test leftendpoint(extents(RL.Rₗᵥ)) == leftendpoint(extents(RL.Rᵥ))
        @test rightendpoint(extents(RL.Rₗᵥ)) > leftendpoint(extents(RL.Rᵥ))
        @test rightendpoint(extents(RL.Rₗᵥ)) ≤ mean(ov)
        @test isapprox(rightendpoint(extents(RL.Rₗᵥ)), mean(ov), atol=σ)

        @test leftendpoint(extents(RL.Rᵥₗ)) > mean(ov)
        @test rightendpoint(extents(RL.Rᵥₗ)) == rightendpoint(extents(RL.Rₗ))

        @test leftendpoint(extents(RL.R̃ₗᵥ)) == leftendpoint(extents(RL.Rₗ))
        @test rightendpoint(extents(RL.R̃ₗᵥ)) > leftendpoint(extents(RL.Rᵥ))
        @test rightendpoint(extents(RL.R̃ₗᵥ)) ≤ mean(ov)
        @test isapprox(rightendpoint(extents(RL.R̃ₗᵥ)), mean(ov), atol=σ)
    end

    RLfd = @atomgrid(2) do
        rₘₐₓ = 300
        ℓₘₐₓ = 40
        rₛₚₗᵢₜ = 25
        overlap = 8
        fd(ρ=0.25)
    end
    test_mixed_gauge(RLfd,25,8,0.25)

    # RLfedvr = @atomgrid(2) do
    #     rₘₐₓ = 300
    #     ℓₘₐₓ = 40
    #     rₛₚₗᵢₜ = 25
    #     overlap = 8
    #     fedvr(N=71,n=10)
    # end
    # test_mixed_gauge(RLfedvr,25,8,300/70)
end
