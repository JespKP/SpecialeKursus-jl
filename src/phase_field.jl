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

        d_qp = function_value(cellvalues_d, qp, d_local)
        ∇d   = function_gradient(cellvalues_d, qp, d_local)

        for i in 1:n_basefuncs
            Nᵢ  = shape_value(cellvalues_d, qp, i)
            ∇Nᵢ = shape_gradient(cellvalues_d, qp, i)

            fe[i] += (g_c / l) * Nᵢ * d_qp * dΩ
            fe[i] += g_c * l * (∇Nᵢ ⋅ ∇d) * dΩ
            fe[i] -= 2 * (1 - d_qp) * H_qp * Nᵢ * dΩ

            for j in 1:n_basefuncs
                Nⱼ  = shape_value(cellvalues_d, qp, j)
                ∇Nⱼ = shape_gradient(cellvalues_d, qp, j)

                ke[i, j] += (g_c / l) * Nᵢ * Nⱼ * dΩ
                ke[i, j] += g_c * l * (∇Nᵢ ⋅ ∇Nⱼ) * dΩ
                ke[i, j] += 2 * H_qp * Nᵢ * Nⱼ * dΩ
            end
        end
    end
end

# Returns the assembled internal force vector f_int BEFORE the constrained
# DOFs are zeroed.  The caller uses this to extract reaction forces at the
# bottom boundary (where f_ext = 0, so f_int = reaction).
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

    # Stores f_int before apply_zero! touches it — this is what we return
    # so the caller can compute reaction forces at Dirichlet boundaries.
    f_int_reaction = zeros(ndofs(dh_u))

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

        # Save f_int BEFORE apply_zero! zeroes the constrained DOFs.
        # At convergence this gives the reaction forces at fixed nodes.
        f_int_reaction .= f_int

        R = f_int
        apply_zero!(R, ch)

        res = norm(R)
        println("  Mechanics iter $iter  ‖R‖ = $res")

        if res ≤ eps_stop
            println("  Mechanics converged at iter $iter")
            converged = true
            break
        end

        apply_zero!(K_u, R, ch)
        u .+= K_u \ (-R)
    end

    if !converged
        @warn "Mechanics solver did not converge after $imax iterations"
    end

    return f_int_reaction
end

function assemble_cell_mechanics!(ke, fe, cellvalues_u, cellvalues_d, u_local, d_local, mat_elastic, mat_fracture)
    k           = mat_fracture.k
    n_basefuncs = getnbasefunctions(cellvalues_u)

    for qp in 1:getnquadpoints(cellvalues_u)
        dΩ   = getdetJdV(cellvalues_u, qp)
        d_qp = function_value(cellvalues_d, qp, d_local)
        g    = (1 - d_qp)^2 + k

        ε    = function_symmetric_gradient(cellvalues_u, qp, u_local)
        σ₀   = mat_elastic.C ⊡ ε

        for i in 1:n_basefuncs
            ∇ˢʸᵐNᵢ = shape_symmetric_gradient(cellvalues_u, qp, i)

            fe[i] += g * (σ₀ ⊡ ∇ˢʸᵐNᵢ) * dΩ

            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues_u, qp, j)
                ke[i, j] += g * (∇ˢʸᵐNᵢ ⊡ mat_elastic.C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
end

"""
    compute_cell_psi(cellvalues_u, cellvalues_d, dh_u, dh_d, u, d, mat_elastic)

Returns a vector of length `n_cells` with the area-averaged undegraded strain
energy density  ψ = ½ ε : C : ε  (same quantity used for the history field H).
Written to VTK so you can see where energy concentrates before fracture.
"""

function compute_cell_H(H, cellvalues_u, dh_u)
    H_cell = zeros(getncells(dh_u.grid))
    n_qp   = getnquadpoints(cellvalues_u)
    for cell in CellIterator(dh_u)
        cid = cellid(cell)
        H_cell[cid] = sum(H[cid, qp] for qp in 1:n_qp) / n_qp
    end
    return H_cell
end

function compute_cell_psi(cellvalues_u, cellvalues_d, dh_u, dh_d, u, d, mat_elastic)
    psi    = zeros(getncells(dh_u.grid))
    dofs_d = zeros(Int, getnbasefunctions(cellvalues_d))

    for cell in CellIterator(dh_u)
        reinit!(cellvalues_u, cell)
        cid     = cellid(cell)
        u_local = u[celldofs(cell)]

        area    = 0.0
        psi_int = 0.0

        for qp in 1:getnquadpoints(cellvalues_u)
            dΩ = getdetJdV(cellvalues_u, qp)
            ε  = function_symmetric_gradient(cellvalues_u, qp, u_local)
            psi_int += 0.5 * (ε ⊡ mat_elastic.C ⊡ ε) * dΩ
            area    += dΩ
        end

        psi[cid] = psi_int / area
    end

    return psi
end

"""
Returns:
  ψ_plus  — positive (tensile) strain energy density
  σ_plus  — tensile stress  ∂ψ⁺/∂ε = λ⟨tr(ε)⟩₊ I + 2μ ε⁺
  σ_minus — compressive stress ∂ψ⁻/∂ε = λ⟨tr(ε)⟩₋ I + 2μ ε⁻
"""

function spectral_split(ε::SymmetricTensor{2,2,T}, λ::Float64, μ::Float64) where T
 
    # --- principal strains and directions ---
    ε11, ε22, ε12 = ε[1,1], ε[2,2], ε[1,2]
    avg   = (ε11 + ε22) / 2
    diff  = (ε11 - ε22) / 2
    R     = sqrt(diff^2 + ε12^2)
 
    ε₁ = avg + R
    ε₂ = avg - R
 
    # principal direction for ε₁
    if abs(R) > 1e-14
        n1x = diff + R
        n1y = ε12
        nrm = sqrt(n1x^2 + n1y^2)
        n1x /= nrm
        n1y /= nrm
    else
        n1x = 1.0
        n1y = 0.0
    end
    n2x = -n1y
    n2y =  n1x
 
    # outer products  nᵢ ⊗ nᵢ
    N1 = SymmetricTensor{2,2}((n1x*n1x, n1x*n1y, n1y*n1y))
    N2 = SymmetricTensor{2,2}((n2x*n2x, n2x*n2y, n2y*n2y))
 
    # positive/negative parts of principal strains
    ε₁⁺ = max(ε₁, 0.0);   ε₁⁻ = min(ε₁, 0.0)
    ε₂⁺ = max(ε₂, 0.0);   ε₂⁻ = min(ε₂, 0.0)
 
    # positive/negative strain tensors
    ε_plus  = ε₁⁺ * N1 + ε₂⁺ * N2
    ε_minus = ε₁⁻ * N1 + ε₂⁻ * N2
 
    # Macaulay brackets on volumetric strain
    tr_plus  = max(tr(ε), 0.0)
    tr_minus = min(tr(ε), 0.0)
 
    # ψ⁺ = λ/2 ⟨tr(ε)⟩²₊ + μ ε⁺:ε⁺
    ψ_plus = 0.5 * λ * tr_plus^2 + μ * (ε_plus ⊡ ε_plus)
 
    # σ⁺ = ∂ψ⁺/∂ε = λ⟨tr(ε)⟩₊ I + 2μ ε⁺
    I2     = one(SymmetricTensor{2,2,Float64})
    σ_plus  = λ * tr_plus  * I2 + 2μ * ε_plus
    σ_minus = λ * tr_minus * I2 + 2μ * ε_minus
 
    return ψ_plus, σ_plus, σ_minus
end
 
"""
Returns the positive and negative algorithmic tangent moduli for the
spectral split.  Used to assemble the consistent stiffness matrix.
"""
function spectral_tangent(ε::SymmetricTensor{2,2,T}, λ::Float64, μ::Float64) where T
 
    ε11, ε22, ε12 = ε[1,1], ε[2,2], ε[1,2]
    avg  = (ε11 + ε22) / 2
    diff = (ε11 - ε22) / 2
    R    = sqrt(diff^2 + ε12^2)
 
    ε₁ = avg + R
    ε₂ = avg - R
 
    if abs(R) > 1e-14
        n1x = diff + R;  n1y = ε12
        nrm = sqrt(n1x^2 + n1y^2)
        n1x /= nrm;  n1y /= nrm
    else
        n1x = 1.0;  n1y = 0.0
    end
    n2x = -n1y;  n2y = n1x
 
    N1 = SymmetricTensor{2,2}((n1x*n1x, n1x*n1y, n1y*n1y))
    N2 = SymmetricTensor{2,2}((n2x*n2x, n2x*n2y, n2y*n2y))
 
    # 4th-order projection tensors  Pᵢ = Nᵢ ⊗ Nᵢ
    P1 = N1 ⊗ N1
    P2 = N2 ⊗ N2
 
    # Heaviside on principal strains
    H1⁺ = ε₁ ≥ 0 ? 1.0 : 0.0;  H1⁻ = 1.0 - H1⁺
    H2⁺ = ε₂ ≥ 0 ? 1.0 : 0.0;  H2⁻ = 1.0 - H2⁺
 
    # Heaviside on volumetric strain
    Hv⁺ = tr(ε) ≥ 0 ? 1.0 : 0.0;  Hv⁻ = 1.0 - Hv⁺
 
    I2   = one(SymmetricTensor{2,2,Float64})
    IxI  = I2 ⊗ I2
 
    C_plus  = λ * Hv⁺ * IxI + 2μ * (H1⁺ * P1 + H2⁺ * P2)
    C_minus = λ * Hv⁻ * IxI + 2μ * (H1⁻ * P1 + H2⁻ * P2)
 
    return C_plus, C_minus
end

function _bottom_x_dofs(dh_u, grid)
    tol          = 1e-10
    bottom_nodes = Set(i for (i, n) in enumerate(grid.nodes) if n.x[2] < tol)
    result       = Int[]
    ndofs_per_nd = ndofs_per_cell(dh_u) ÷ length(first(CellIterator(dh_u)).nodes)
    for cell in CellIterator(dh_u)
        dofs  = celldofs(cell)
        nodes = cell.nodes
        for (i, node) in enumerate(nodes)
            if node ∈ bottom_nodes
                push!(result, dofs[(i - 1) * ndofs_per_nd + 1])   # component 1 = x
            end
        end
    end
    return unique!(sort!(result))
end
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Mechanics solver using spectral split
# ─────────────────────────────────────────────────────────────────────────────
 
function solve_mechanics_spectral!(K_u, f_u, u, cellvalues_u, cellvalues_d,
                                   dh_u, dh_d, d, mat_elastic, mat_fracture, ch;
                                   imax::Int         = 20,
                                   eps_stop::Float64 = 1e-6)
 
    n_basefuncs_u = getnbasefunctions(cellvalues_u)
    n_basefuncs_d = getnbasefunctions(cellvalues_d)
    ke      = zeros(n_basefuncs_u, n_basefuncs_u)
    fe      = zeros(n_basefuncs_u)
    f_int   = zeros(ndofs(dh_u))
    d_local = zeros(n_basefuncs_d)
    u_local = zeros(n_basefuncs_u)
    dofs_d  = zeros(Int, n_basefuncs_d)
 
    f_int_reaction = zeros(ndofs(dh_u))
 
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
 
            assemble_cell_mechanics_spectral!(ke, fe, cellvalues_u, cellvalues_d,
                                              u_local, d_local, mat_elastic, mat_fracture)
            assemble!(assembler, celldofs(cell_u), ke, fe)
        end
 
        f_int_reaction .= f_int
 
        R = f_int
        apply_zero!(R, ch)
 
        res = norm(R) / length(R)
        println("  Mechanics iter $iter  ‖R‖/n = $res")
 
        if res ≤ eps_stop
            println("  Mechanics converged at iter $iter")
            converged = true
            break
        end
 
        apply_zero!(K_u, R, ch)
        u .+= K_u \ (-R)
    end
 
    if !converged
        @warn "Mechanics solver did not converge after $imax iterations"
    end
 
    return f_int_reaction
end

function update_history_spectral!(H, cellvalues_u, dh_u, u, mat_elastic)
    λ = mat_elastic.C[1,1,2,2]         # C_1122 = λ for isotropic plane strain
    μ = mat_elastic.C[1,2,1,2]         # C_1212 = μ
 
    for cell in CellIterator(dh_u)
        reinit!(cellvalues_u, cell)
        cid     = cellid(cell)
        u_local = u[celldofs(cell)]
 
        for qp in 1:getnquadpoints(cellvalues_u)
            ε            = function_symmetric_gradient(cellvalues_u, qp, u_local)
            ψ_plus, _, _ = spectral_split(ε, λ, μ)
            H[cid, qp]   = max(H[cid, qp], ψ_plus)
        end
    end
end


function assemble_cell_mechanics_spectral!(ke, fe, cellvalues_u, cellvalues_d,
                                           u_local, d_local, mat_elastic, mat_fracture)
    k           = mat_fracture.k
    λ           = mat_elastic.C[1,1,2,2]
    μ           = mat_elastic.C[1,2,1,2]
    n_basefuncs = getnbasefunctions(cellvalues_u)
 
    for qp in 1:getnquadpoints(cellvalues_u)
        dΩ   = getdetJdV(cellvalues_u, qp)
        d_qp = function_value(cellvalues_d, qp, d_local)
        g    = (1 - d_qp)^2 + k
 
        ε                      = function_symmetric_gradient(cellvalues_u, qp, u_local)
        _, σ_plus, σ_minus     = spectral_split(ε, λ, μ)
        C_plus, C_minus        = spectral_tangent(ε, λ, μ)
 
        # degraded stress: g(d)·σ⁺ + σ⁻
        σ_deg = g * σ_plus + σ_minus
 
        for i in 1:n_basefuncs
            ∇ˢʸᵐNᵢ = shape_symmetric_gradient(cellvalues_u, qp, i)
            fe[i]  += (σ_deg ⊡ ∇ˢʸᵐNᵢ) * dΩ
 
            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues_u, qp, j)
                # consistent tangent: g(d)·C⁺ + C⁻
                C_deg      = g * C_plus + C_minus
                ke[i, j]  += (∇ˢʸᵐNᵢ ⊡ C_deg ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
end
 
function compute_cell_psi_spectral(cellvalues_u, cellvalues_d, dh_u, dh_d, u, d, mat_elastic)
    λ   = mat_elastic.C[1,1,2,2]
    μ   = mat_elastic.C[1,2,1,2]
    psi = zeros(getncells(dh_u.grid))
 
    for cell in CellIterator(dh_u)
        reinit!(cellvalues_u, cell)
        cid     = cellid(cell)
        u_local = u[celldofs(cell)]
        area    = 0.0
        psi_int = 0.0
 
        for qp in 1:getnquadpoints(cellvalues_u)
            dΩ           = getdetJdV(cellvalues_u, qp)
            ε            = function_symmetric_gradient(cellvalues_u, qp, u_local)
            ψ_plus, _, _ = spectral_split(ε, λ, μ)
            psi_int     += ψ_plus * dΩ
            area        += dΩ
        end
 
        psi[cid] = psi_int / area
    end
 
    return psi
end