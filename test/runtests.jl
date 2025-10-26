using Graphs
using LinearAlgebra
using SemiringLU
using SparseArrays
using Test
using TropicalNumbers

@testset "(ℝ ∪ {+∞}, +, ×, 0, 1)" begin
    @testset "Generic Inference" begin
        #=
        # Section 6.7
        A = [
            0.0 0.6 0.4
            0.0 0.1 0.9
            0.0 1.0 0.0
        ]

        @test sinv(A) == [
            1.0 0.0 0.0
            Inf Inf Inf
            Inf Inf Inf     
        ]
        =#
    end

    @testset "random matrices" begin
        m = 1000; n = 5

        for _ in 1:10
            A = rand(m, m)
            B = rand(m, n)
            b = rand(m)

            # inversion
            @test sinv(A) ≈ inv(I - A)

            # left division
            @test sldiv!(A, copy(B)) ≈ (I - A) \ B
            @test sldiv!(A, copy(b)) ≈ (I - A) \ b

            # right division
            @test srdiv!(copy(B)', A) ≈ B' / (I - A)
            @test srdiv!(copy(b)', A) ≈ b' / (I - A)
        end
    end
end

@testset "(ℝ ∪ {-∞, +∞}, ∧, +, +∞, 0)" begin
    @testset "Generic Inference" begin
        # Example 6.3
        A = TropicalMinPlusF64[
            Inf 9.0 8.0 Inf
            Inf Inf 6.0 Inf
            Inf Inf Inf 7.0
            5.0 Inf Inf Inf
        ]

        @test sinv(A) == TropicalMinPlusF64[
             0.0  9.0  8.0 15.0
            18.0  0.0  6.0 13.0
            12.0 21.0  0.0  7.0
             5.0 14.0 13.0  0.0
        ]

        # Example 6.8
        A = TropicalMinPlusF64[
            Inf 7.0 1.0
            4.0 Inf Inf
            Inf 2.0 Inf
        ]

        @test sinv(A) == TropicalMinPlusF64[
            0.0 3.0 1.0
            4.0 0.0 5.0
            6.0 2.0 0.0     
        ]
    end

    @testset "random matrices" begin
        n = 1000; p = 0.01

        for _ in 1:10
            m0 = sprand(n, n, p)
            m1 = Matrix(m0); g1 = DiGraph(m0)
            m2 = zeros(TropicalMinPlusF64, size(m0))

            for j in axes(m0, 2)
                for p in nzrange(m0, j)
                    i = rowvals(m0)[p]
                    v = nonzeros(m0)[p]

                    m2[i, j] = v
                end
            end

            D1 = floyd_warshall_shortest_paths(g1, m1)
            D2 = sinv(m2)

            # inversion
            @test TropicalMinPlusF64.(D1.dists) ≈ D2
        end
    end
end

@testset "([0, 1], ∨, ×, 0, 1)" begin
    @testset "Generic Inference" begin
        # Section 6.7
        A = TropicalMaxMulF64[
            0.0 0.6 0.4
            0.0 0.1 0.9
            0.0 1.0 0.0
        ]

        @test sinv(A) == TropicalMaxMulF64[
            1.0 0.6 0.54
            0.0 1.0 0.9
            0.0 1.0 1.0     
        ]
    end
end
