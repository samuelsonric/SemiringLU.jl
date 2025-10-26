function sinv(x::TropicalAndOr)
    return TropicalAndOr(true)
end

function sinv(x::T) where {T <: AbstractFloat}
    if isone(x)
        n = posinf(T)
    else
        n = inv(one(T) - x)
    end

    return n
end

function sinv(x::TropicalMaxMul{T}) where {T}
    if x.n > one(T)
        n = posinf(T)
    else
        n = one(T)
    end

    return TropicalMaxMul(n)
end

function sinv(x::TropicalMaxPlus{T}) where {T}
    if x.n > zero(T)
        n = posinf(T)
    else
        n = zero(T)
    end

    return TropicalMaxPlus(n)
end
    
function sinv(x::TropicalMinPlus{T}) where {T}
    if x.n < zero(T)
        n = neginf(T)
    else
        n = zero(T)
    end

    return TropicalMinPlus(n)
end
