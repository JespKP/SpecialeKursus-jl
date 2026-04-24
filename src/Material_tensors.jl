
function plane_strain_tensors(Emod, nu)
    lambda = Emod * nu / ((1 + nu) * (1 - 2 * nu))
    Gmod   = Emod / (2 * (1 + nu))

    # Correct: σ = λ·tr(ε)·I + 2G·ε
    C     = gradient(eps -> lambda * tr(eps) * one(eps) + 2 * Gmod * eps,
                     zero(SymmetricTensor{2, 2}))
    C_dil = gradient(eps -> lambda * tr(eps) * one(eps),
                     zero(SymmetricTensor{2, 2}))
    C_dev = C - C_dil   # = 2G·I_sym (symmetric 4th-order identity times 2G)

    return LinearElasticMaterial(C, C_dil, C_dev)
end

function plane_stress_tensors(Emod, nu)
    Gmod = Emod / (2 * (1 + nu))
    λ = Emod * nu / (1 - nu^2)
    C = gradient(eps -> λ * tr(eps) * one(eps) + 2 * Gmod * eps,
                 zero(SymmetricTensor{2, 2}))
    return LinearElasticMaterial(C, C, C)
end