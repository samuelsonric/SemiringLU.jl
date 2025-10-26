using LinearAlgebra
using SemiringLU
using Test
using TropicalNumbers

@testset "(ℝ, +, ×, 0, 1)" begin
    m = 1000
    n = 5

    for _ in 1:10
        A = rand(m, m)
        B = rand(m, n)
        b = rand(n)

        # inversion
        @test sinv(A) ≈ inv(I - A)

        # left division
        @test sldiv!(A, copy(B)) ≈ ldiv!(I - A, copy(B))
        @test sldiv!(A, copy(b)) ≈ ldiv!(I - A, copy(b))

        # right division
        @test srdiv!(copy(B)', M) ≈ rdiv!(copy(B)', I - A)
        @test srdiv!(copy(b)', M) ≈ rdiv!(copy(b)', I - A)
    end
end

