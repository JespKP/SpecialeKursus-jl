function assemble_plastic!(K, f_int, trial_states, dh, scheme, mat::LinearElasticMaterial,
                            committed_states, u_trial, u_committed, h)
    n_basefuncs = getnbasefunctions(get_primary_cv(scheme))
    ke     = zeros(n_basefuncs, n_basefuncs)
    fe_int = zeros(n_basefuncs)
    assembler = start_assemble(K)

    for cell in CellIterator(dh)
    reinit_scheme!(scheme, cell)
    fill!(ke, 0.0)
    fill!(fe_int, 0.0)

    cid     = cellid(cell)
    dofs    = celldofs(cell)
    Δu_cell = u_trial[dofs] .- u_committed[dofs]

    assemble_cell_plastic!(ke, fe_int, scheme, mat,
                            committed_states[cid], trial_states[cid],
                            Δu_cell, h)

    assemble!(assembler, dofs, ke)
    assemble!(f_int, dofs, fe_int)
end

    return K, f_int, trial_states
end

function assemble_cell_plastic!(ke, fe_int, scheme, mat::LinearElasticMaterial,
                                 committed_cell_states, trial_cell_states,
                                 Δu_cell, h)

    n_basefuncs = getnbasefunctions(get_primary_cv(scheme))

    for qp in 1:getnquadpoints(get_primary_cv(scheme))
        dΩ = getdetJdV(get_primary_cv(scheme), qp)
        Δε = function_symmetric_gradient(get_primary_cv(scheme), qp, Δu_cell)

        σ, C, new_state = material_response(Δε, committed_cell_states[qp], mat, h)
        trial_cell_states[qp] = new_state

        for i in 1:n_basefuncs
            ∇ˢʸᵐNᵢ = shape_symmetric_gradient(get_primary_cv(scheme), qp, i)
            fe_int[i] += (∇ˢʸᵐNᵢ ⊡ σ) * dΩ
            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(get_primary_cv(scheme), qp, j)
                ke[i, j] += (∇ˢʸᵐNᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
end