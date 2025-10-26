struct SemiringLU{T, M <: AbstractMatrix{T}}
    factors::M
end

function Base.getproperty(F::SemiringLU, d::Symbol)
    if d == :L
        p = StrictLowerTriangular(F.factors)
    elseif d === :U
        p = UpperTriangular(F.factors)
    else
        p = getfield(F, d)
    end

    return p
end

function Base.show(io::IO, mime::MIME"text/plain", F::T) where {T <: SemiringLU}
    print(io, "$T:")
    print(io, "\nL factor:\n")
    show(io, mime, F.L)
    print(io, "\nU factor:\n")
    show(io, mime, F.U)
    return
end

function slu!(A::AbstractMatrix)
    sgetrf!(A)
    return SemiringLU(A)
end

function slu(A::AbstractMatrix)
    return slu!(Matrix(A))
end

function sldiv!(C::AbstractArray, A, B::AbstractArray)
    return sldiv!(A, copyto!(C, B))
end

function sldiv!(A::Number, B::AbstractArray)
    @. B = sinv(A) * B
    return B
end

function sldiv!(A::StrictLowerTriangular, B::AbstractArray)
    strmm!(parent(A), B, Val(:L), Val(:L))
    return B
end

function sldiv!(A::UpperTriangular, B::AbstractArray)
    strmm!(parent(A), B, Val(:U), Val(:L))
    return B
end

function sldiv!(A::SemiringLU, B::AbstractArray)
    strmm!(A.factors, B, Val(:L), Val(:L))
    strmm!(A.factors, B, Val(:U), Val(:L))
    return B
end

function sldiv!(A::AbstractMatrix, B::AbstractArray)
    return sldiv!(slu(A), B)
end

function srdiv!(C::AbstractArray, B::AbstractArray, A)
    return srdiv!(copyto!(C, B), A)
end

function srdiv!(B::AbstractArray, A::Number)
    B .*= sinv(A)
    return B
end

function srdiv!(B::AbstractArray, A::StrictLowerTriangular)
    strmm!(parent(A), B, Val(:L), Val(:R))
    return B
end

function srdiv!(B::AbstractArray, A::UpperTriangular)
    strmm!(parent(A), B, Val(:U), Val(:R))
    return B
end

function srdiv!(B::AbstractArray, A::SemiringLU)
    strmm!(A.factors, B, Val(:U), Val(:R))
    strmm!(A.factors, B, Val(:L), Val(:R))
    return B
end

function sinv(A::AbstractMatrix{T}) where {T}
    B = zeros(T, size(A))
    B[diagind(B)] .= one(T)
    return sldiv!(A, B)
end

function sgetrf2!(A::AbstractMatrix{T}) where {T}
    @assert size(A, 1) == size(A, 2)
    
    @inbounds for i in axes(A, 1)
        #
        #   A = [ Aii Ain ]
        #       [ Ani Ann ]
        #
        Aii = A[i, i]
        Ain = @view A[i,         i + 1:end]
        Ani = @view A[i + 1:end, i]
        Ann = @view A[i + 1:end, i + 1:end]

        #
        #   Ani ← Ani Aii*
        # 
        srdiv!(Ani, Aii)

        #
        #   Ann ← Ann + Ani Ain
        #
        mul!(Ann, Ani, Ain', one(T), one(T))
    end

    return
end

function sgetrf!(A::AbstractMatrix{T}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2)

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1

        #
        #   A = [ Abb Abn ]
        #       [ Anb Ann ]
        #
        Abb = @view A[strt:stop, strt:stop]

        #
        #   Abb ← Lbb + Ubb
        # 
        sgetrf2!(Abb)

        if stop < n
            Abn = @view A[strt:stop,  stop + 1:n]
            Anb = @view A[stop + 1:n, strt:stop]
            Ann = @view A[stop + 1:n, stop + 1:n]

            # 
            #   Abn ← Lbb* Abn
            #    
            strmm!(Abb, Abn, Val(:L), Val(:L))

            #
            #   Anb ← Anb Ubb*
            #
            strmm!(Abb, Anb, Val(:U), Val(:R))

            #
            #   Ann ← Ann + Anb Abn
            #
            mul!(Ann, Anb, Abn, one(T), one(T))
        end
    end

    return
end

function strmm2!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:L}, side::Val{:L}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    @inbounds for i in axes(A, 1)
        #
        #   A = [ 0   0   ]
        #       [ Ani Ann ]
        #
        Ani = @view A[i + 1:end, i]

        #
        #   B = [ Bi ]
        #       [ Bn ]
        #
        Bi = @view B[i,         :]
        Bn = @view B[i + 1:end, :]

        #
        #   Bn ← Bn + Ani Bi
        #
        mul!(Bn, Ani, Bi |> transpose, one(T), one(T))
    end

    return
end

function strmm!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:L}, side::Val{:L}, blocksize = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1

        #
        #   A = [ Abb   0   ]
        #       [ Anb   Ann ]
        #
        Abb = @view A[strt:stop, strt:stop]

        #
        #   B = [ Bb ]
        #       [ Bn ]
        #
        Bb = @view B[strt:stop, :]

        #
        #   Bb ← Abb* Bb
        #
        strmm2!(Abb, Bb, uplo, side)

        if stop < n
            Anb = @view A[stop + 1:end, strt:stop]
            Bn  = @view B[stop + 1:end, :]

            #
            #
            #   Bn ← Bn + Anb Bb
            #
            mul!(Bn, Anb, Bb, one(T), one(T))
        end
    end

    return
end

function strmm2!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:U}, side::Val{:L}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    @inbounds for i in reverse(axes(A, 1))
        #
        #   A = [ Ann Ani ]
        #       [ 0   Aii ]
        #
        Aii = A[i, i]
        Ani = @view A[begin:i - 1, i]

        #
        #   B = [ Bn ]
        #       [ Bi ]
        #
        Bi = @view B[i,           :]
        Bn = @view B[begin:i - 1, :]

        #
        #   Bi ← Aii* Bi
        #    
        sldiv!(Aii, Bi)

        #
        #   Bn ← Bn + Ani Bi
        #
        mul!(Bn, Ani, Bi |> transpose, one(T), one(T))
    end

    return
end

function strmm!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:U}, side::Val{:L}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1)

    n = size(A, 1)

    @inbounds for stop in n:-blocksize:1
        size = min(blocksize, stop)
        strt = stop - size + 1

        #
        #   A = [ Ann Anb ]
        #       [ 0   Abb ]
        #
        Abb = @view A[strt:stop, strt:stop]

        #
        #   B = [ Bn ]
        #       [ Bb ]
        #
        Bb = @view B[strt:stop, :]

        #
        #   Bb ← Abb* Bbb
        #
        strmm2!(Abb, Bb, uplo, side)

        if 1 < strt
            Anb = @view A[begin:strt - 1, strt:stop]
            Bn  = @view B[begin:strt - 1, :]

            #
            #   Bn ← Bn + Anb Bb
            #
            mul!(Bn, Anb, Bb, one(T), one(T))
        end
    end

    return
end

function strmm2!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:L}, side::Val{:R}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 2)

    @inbounds for i in reverse(axes(A, 1))
        #
        #   A = [ Ann 0 ]
        #       [ Ain 0 ]
        #
        Ain = @view A[i, begin:i - 1]

        #
        #   B = [ Bn Bi ]
        #
        Bi = @view B[:, i]
        Bn = @view B[:, begin:i - 1]
 
        #
        #
        #   Bn ← Bn + Bi Ain
        #
        mul!(Bn, Bi, Ain |> transpose, one(T), one(T))
    end

    return
end

function strmm!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:L}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 2)

    n = size(A, 1)

    @inbounds for stop in n:-blocksize:1
        size = min(blocksize, stop)
        strt = stop - size + 1

        #
        #   A = [ Ann 0   ]
        #       [ Abn Abb ]
        #
        Abb = @view A[strt:stop, strt:stop]

        #
        #   B = [ Bn Bb ]
        #
        Bb = @view B[:, strt:stop]

        #
        #   Bb ← Bb Abb*
        #
        strmm2!(Abb, Bb, uplo, side)

        if strt > 1
            Abn = @view A[strt:stop, begin:strt - 1]
            Bn  = @view B[:,         begin:strt - 1]
     
            #
            #
            #   Bn ← Bn + Bb Abn
            #
            mul!(Bn, Bb, Abn, one(T), one(T))
        end
    end

    return
end

function strmm2!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:U}, side::Val{:R}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 2)

    @inbounds for i in axes(A, 1)
        #
        #   A = [ Aii Ain ]
        #       [ 0   Ann ]
        #
        Aii = A[i, i]
        Ain = @view A[i, i + 1:end]
 
        #
        #   B = [ Bi Bn ]
        #
        Bi = @view B[:, i]
        Bn = @view B[:, i + 1:end]

        #
        #   Bi ← Bi Aii*
        #
        srdiv!(Bi, Aii)
 
        #
        #
        #   Bn ← Bn + Bi Ain
        #
        mul!(Bn, Bi, Ain |> transpose, one(T), one(T))
    end
end

function strmm!(A::AbstractMatrix{T}, B::AbstractArray{T}, uplo::Val{:U}, side::Val{:R}, blocksize::Int = DEFAULT_BLOCK_SIZE) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 2)

    n = size(A, 1)

    @inbounds for strt in 1:blocksize:n
        size = min(blocksize, n - strt + 1)
        stop = strt + size - 1

        #
        #   A = [ Abb Abn ]
        #       [ 0   Ann ]
        #
        Abb = @view A[strt:stop, strt:stop]
 
        #
        #   B = [ Bb Bn ]
        #
        Bb = @view B[:, strt:stop]

        #
        #   Bb ← Bb Abb*
        #
        strmm2!(Abb, Bb, uplo, side)
 
        if stop < n
            Abn = @view A[strt:stop, stop + 1:end]
            Bn  = @view B[:,         stop + 1:end]

            #
            #
            #   Bn ← Bn + Bb Abn
            #
            mul!(Bn, Bb, Abn, one(T), one(T))
        end
    end

    return
end
