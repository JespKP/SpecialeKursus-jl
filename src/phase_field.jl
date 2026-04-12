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

function solve_phasefield!(K_d, f_d, cellvalues_d, dh_d, d, H, mat_fracture;
                           imax::Int         = 20,
                           eps_stop::Float64 = 1e-6)

    n_basefuncs = getnbasefunctions(cellvalues_d)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    converged = false

    for iter in 1:imax

        fill!(K_d.nzval, 0.0)
        fill!(f_d, 0.0)
        assembler = start_assemble(K_d, f_d)

        for cell in CellIterator(dh_d)
            reinit!(cellvalues_d, cell)
            cid     = cellid(cell)
            dofs    = celldofs(cell)
            d_local = d[dofs]

            fill!(ke, 0.0)
            fill!(fe, 0.0)

            assemble_cell_phasefield!(ke, fe, cellvalues_d, d_local, H[cid, :], mat_fracture)
            assemble!(assembler, dofs, ke, fe)
        end

        res = norm(f_d)
        println("  Phase field iter $iter  ‖R‖ = $res")

        if res ≤ eps_stop
            println("  Phase field converged at iter $iter")
            converged = true
            break
        end

        d .+= K_d \ (-f_d)

    end

    if !converged
        @warn "Phase field solver did not converge after $imax iterations"
    end

    return d
end

function assemble_cell_phasefield!(ke, fe, cellvalues_d, d_local, H_cell, mat_fracture)
    g_c         = mat_fracture.g_c
    l           = mat_fracture.l
    n_basefuncs = getnbasefunctions(cellvalues_d)

    for qp in 1:getnquadpoints(cellvalues_d)
        dΩ   = getdetJdV(cellvalues_d, qp)
        H_qp = H_cell[qp]

        # Interpolate current d at this quadrature point
        d_qp = function_value(cellvalues_d, qp, d_local)
        ∇d   = function_gradient(cellvalues_d, qp, d_local)

        for i in 1:n_basefuncs
            Nᵢ  = shape_value(cellvalues_d, qp, i)
            ∇Nᵢ = shape_gradient(cellvalues_d, qp, i)

            # Residual — all three terms of the weak form evaluated at current d
            fe[i] += (g_c / l) * Nᵢ * d_qp * dΩ          # term 1
            fe[i] += g_c * l * (∇Nᵢ ⋅ ∇d) * dΩ           # term 2
            fe[i] -= 2 * (1 - d_qp) * H_qp * Nᵢ * dΩ     # term 3

            for j in 1:n_basefuncs
                Nⱼ  = shape_value(cellvalues_d, qp, j)
                ∇Nⱼ = shape_gradient(cellvalues_d, qp, j)

                # Tangent — derivative of residual w.r.t. d
                ke[i, j] += (g_c / l) * Nᵢ * Nⱼ * dΩ        # from term 1
                ke[i, j] += g_c * l * (∇Nᵢ ⋅ ∇Nⱼ) * dΩ      # from term 2
                ke[i, j] += 2 * H_qp * Nᵢ * Nⱼ * dΩ         # from term 3
            end
        end
    end
end

function solve_mechanics!(K_u, f_u, u, cellvalues_u, cellvalues_d, dh_u, dh_d, d, mat_elastic, mat_fracture, ch;
                          imax::Int         = 20,
                          eps_stop::Float64 = 1e-10)

    n_basefuncs_u = getnbasefunctions(cellvalues_u)
    n_basefuncs_d = getnbasefunctions(cellvalues_d)
    ke      = zeros(n_basefuncs_u, n_basefuncs_u)
    fe      = zeros(n_basefuncs_u)
    f_int   = zeros(ndofs(dh_u))
    d_local = zeros(n_basefuncs_d)
    u_local = zeros(n_basefuncs_u)
    dofs_d  = zeros(Int, n_basefuncs_d)

    # Initialize u with prescribed displacements before Newton loop
    apply!(u, ch)

    converged = false

    for iter in 1:imax

        fill!(K_u.nzval, 0.0)
        fill!(f_int, 0.0)
        assembler = start_assemble(K_u, f_int)

        for cell_u in CellIterator(dh_u)
            reinit!(cellvalues_u, cell_u)
            reinit!(cellvalues_d, cell_u)

            cid = cellid(cell_u)
            celldofs!(dofs_d, dh_d, cid)
            d_local .= d[dofs_d]
            u_local .= u[celldofs(cell_u)]

            fill!(ke, 0.0)
            fill!(fe, 0.0)

            assemble_cell_mechanics!(ke, fe, cellvalues_u, cellvalues_d, u_local, d_local, mat_elastic, mat_fracture)
            assemble!(assembler, celldofs(cell_u), ke, fe)
        end

        # Residual R = f_int (no external traction)
        R = f_int
        apply_zero!(R, ch)   # zero out constrained DOFs in residual

        res = norm(R)
        println("  Mechanics iter $iter  ‖R‖ = $res")

        if res ≤ eps_stop
            println("  Mechanics converged at iter $iter")
            converged = true
            break
        end

        # Newton correction — Δu = 0 on constrained DOFs
        apply_zero!(K_u, R, ch)
        u .+= K_u \ (-R)
    end

    if !converged
        @warn "Mechanics solver did not converge after $imax iterations"
    end
end

function assemble_cell_mechanics!(ke, fe, cellvalues_u, cellvalues_d, u_local, d_local, mat_elastic, mat_fracture)
    k           = mat_fracture.k
    n_basefuncs = getnbasefunctions(cellvalues_u)

    for qp in 1:getnquadpoints(cellvalues_u)
        dΩ   = getdetJdV(cellvalues_u, qp)
        d_qp = function_value(cellvalues_d, qp, d_local)
        g    = (1 - d_qp)^2 + k

        # Compute strain and stress at current u
        ε    = function_symmetric_gradient(cellvalues_u, qp, u_local)
        σ₀   = mat_elastic.C ⊡ ε

        for i in 1:n_basefuncs
            ∇ˢʸᵐNᵢ = shape_symmetric_gradient(cellvalues_u, qp, i)

            # Internal force: ∫ g(d) σ₀(u) : ∇ˢʸᵐNᵢ dV
            fe[i] += g * (σ₀ ⊡ ∇ˢʸᵐNᵢ) * dΩ

            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues_u, qp, j)

                # Tangent: ∂(f_int)/∂u = ∫ g(d) ∇ˢʸᵐNᵢ : C : ∇ˢʸᵐNⱼ dV
                ke[i, j] += g * (∇ˢʸᵐNᵢ ⊡ mat_elastic.C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
end