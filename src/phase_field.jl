function update_history!(H, cellvalues_u, dh_u, u, mat_elastic)
    for cell in CellIterator(dh_u)
        reinit!(cellvalues_u, cell)
        cid     = cellid(cell)
        u_local = u[celldofs(cell)]

        for qp in 1:getnquadpoints(cellvalues_u)
            ε      = function_symmetric_gradient(cellvalues_u, qp, u_local)
            ψ_plus = 0.5 * (ε ⊡ mat_elastic.C ⊡ ε)
            H[cid, qp] = max(H[cid, qp], ψ_plus)
        end
    end
end

function solve_phasefield!(K_d, f_d, cellvalues_d, dh_d, H, mat_fracture)
    assembler   = start_assemble(K_d, f_d)
    n_basefuncs = getnbasefunctions(cellvalues_d)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    for cell in CellIterator(dh_d)
        reinit!(cellvalues_d, cell)
        cid  = cellid(cell)
        dofs = celldofs(cell)

        fill!(ke, 0.0)
        fill!(fe, 0.0)

        assemble_cell_phasefield!(ke, fe, cellvalues_d, H[cid, :], mat_fracture)
        assemble!(assembler, dofs, ke, fe)
    end

    d = K_d \ f_d
    return d
end

function assemble_cell_phasefield!(ke, fe, cellvalues_d, H_cell, mat_fracture)
    g_c         = mat_fracture.g_c
    l           = mat_fracture.l
    n_basefuncs = getnbasefunctions(cellvalues_d)

    for qp in 1:getnquadpoints(cellvalues_d)
        dΩ   = getdetJdV(cellvalues_d, qp)
        H_qp = H_cell[qp]

        for i in 1:n_basefuncs
            Nᵢ  = shape_value(cellvalues_d, qp, i)
            ∇Nᵢ = shape_gradient(cellvalues_d, qp, i)

            fe[i] += 2 * H_qp * Nᵢ * dΩ

            for j in 1:n_basefuncs
                Nⱼ  = shape_value(cellvalues_d, qp, j)
                ∇Nⱼ = shape_gradient(cellvalues_d, qp, j)

                ke[i, j] += (g_c / l + 2 * H_qp) * Nᵢ * Nⱼ * dΩ
                ke[i, j] += g_c * l * (∇Nᵢ ⋅ ∇Nⱼ) * dΩ
            end
        end
    end
end

function solve_mechanics!(K_u, f_u, cellvalues_u, cellvalues_d, dh_u, dh_d, d, mat_elastic, mat_fracture, ch)
    assembler     = start_assemble(K_u, f_u)
    n_basefuncs_u = getnbasefunctions(cellvalues_u)
    n_basefuncs_d = getnbasefunctions(cellvalues_d)
    ke      = zeros(n_basefuncs_u, n_basefuncs_u)
    d_local = zeros(n_basefuncs_d)


    dofs_d = zeros(Int, n_basefuncs_d)
for cell_u in CellIterator(dh_u)
    reinit!(cellvalues_u, cell_u)
    cid = cellid(cell_u)
    # Get d dofs for this cell directly
    celldofs!(dofs_d, dh_d, cid)
    d_local .= d[dofs_d]

        reinit!(cellvalues_d, cell_u)

        fill!(ke, 0.0)
        assemble_cell_mechanics!(ke, cellvalues_u, cellvalues_d, d_local, mat_elastic, mat_fracture)
        assemble!(assembler, celldofs(cell_u), ke)
    end

    fill!(f_u, 0.0)
    apply!(K_u, f_u, ch)
    u = K_u \ f_u
    return u
end

function assemble_cell_mechanics!(ke, cellvalues_u, cellvalues_d, d_local, mat_elastic, mat_fracture)
    k           = mat_fracture.k
    n_basefuncs = getnbasefunctions(cellvalues_u)

    for qp in 1:getnquadpoints(cellvalues_u)
        dΩ = getdetJdV(cellvalues_u, qp)

        # Use cellvalues_d to interpolate d at this quadrature point
        # cellvalues_u would be wrong here — d is a scalar field on ip_d
        d_qp = function_value(cellvalues_d, qp, d_local)
        g    = (1 - d_qp)^2 + k

        for i in 1:n_basefuncs
            ∇ˢʸᵐNᵢ = shape_symmetric_gradient(cellvalues_u, qp, i)
            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues_u, qp, j)
                ke[i, j] += g * (∇ˢʸᵐNᵢ ⊡ mat_elastic.C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
end