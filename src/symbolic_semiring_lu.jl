struct SymbolicSemiringLU{I}
    ord::FVector{I}
    res::BipartiteGraph{I, I, FVector{I}, OneTo{I}}
    sep::BipartiteGraph{I, I, FVector{I}, FVector{I}}
    rel::BipartiteGraph{I, I, FVector{I}, FVector{I}}
    chd::BipartiteGraph{I, I, FVector{I}, FVector{I}}
    nMptr::I
    nMval::I
    nNval::I
    nRval::I
    nLval::I
    nFval::I
end

function Base.show(io::IO, ::MIME"text/plain", symb::T) where {T <: SymbolicSemiringLU}
    frt = symb.nFval
    nnz = symb.nRval + symb.nLval + symb.nLval

    print(io, "$T:")
    print(io, "\n  maximum front-size: $frt")
    print(io, "\n  Lnz + Unz: $nnz")
end

function SymbolicSemiringLU(matrix::SparseMatrixCSC; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM, snd::SupernodeType = DEFAULT_SUPERNODE_TYPE)
    return SymbolicSemiringLU(matrix, alg, snd)
end

function SymbolicSemiringLU(matrix::SparseMatrixCSC{<:Any, I}, alg::PermutationOrAlgorithm, snd::SupernodeType) where {I <: Integer}
    # `digraph` is a directed graph 
    #
    #   D = (V, A)
    #
    digraph = BipartiteGraph(matrix)

    # `graph` is an undirected graph
    #
    #   G = (V, E),
    #
    # where E is the set
    #
    #   E := { {v, w} | (v, x) ∈ A and (w, x) ∈ A }.
    #
    graph = linegraph(digraph, reverse(digraph)) 

    # `tree` is a tree decomposition T of G with
    # perfect elimination ordering `perm`
    perm, tree = cliquetree(graph, alg, snd)    
    res = residuals(tree)
    sep = separators(tree)

    reltgt = FVector{I}(undef, ne(sep))
    rel = BipartiteGraph(nov(sep), nv(sep), ne(sep), pointers(sep), reltgt)

    #
    #   nMptr = | Mptr |
    #
    nMptr = jMptr = one(I)

    #
    #   nMval = | Mval |
    #
    nMval = jMval = zero(I)

    #
    #   nNval = | nVal |
    #
    nNval = jNval = zero(I)

    #
    #   nFval = | Fval |
    #
    nFval = zero(I)

    #
    #   nRval = | Rval |
    #
    nRval = zero(I)

    #
    #   nLval = | Lval |
    #
    nLval = zero(I)

    # j is a node in T
    for j in vertices(res)
        #
        #   nn = | res(j) |
        #
        nn = eltypedegree(res, j)

        #
        #   na = | sep(j) |
        #
        na = eltypedegree(sep, j)

        #
        #   mj = | bag(j) |
        #
        nj = nn + na

        #
        #   bag(j)
        #
        bag = tree[j]

        # i is a child node of j
        for i in childindices(tree, j)
            # write relative indices to rel(i)
            pj = one(I); vj = bag[pj]

            for pi in incident(sep, i)
                vi = targets(sep)[pi]

                while vj < vi
                    pj += one(I); vj = bag[pj]
                end

                targets(rel)[pi] = pj
            end

            #
            #   ma = | sep(i) |
            #
            ma = eltypedegree(sep, i)

            jMptr -= one(I)
            jMval -= ma * ma
            jNval -= ma
        end

        if ispositive(na)
            jMptr += one(I)
            jMval += na * na
            jNval += na

            nMptr = max(nMptr, jMptr)
            nMval = max(nMval, jMval)
            nNval = max(nNval, jNval)
        end

        nFval = max(nFval, nj)
        nRval += nn * nn
        nLval += nn * na
    end

    ord = FVector{I}(perm)
    chd = tree.tree.tree.graph

    return SymbolicSemiringLU(ord, res, sep, rel, chd, nMptr, nMval, nNval, nRval, nLval, nFval)
end
