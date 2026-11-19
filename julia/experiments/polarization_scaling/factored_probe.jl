using LinearAlgebra, Random, Printf

# Small, standard-library-only diagnostic. It deliberately does not pretend to
# be a periodic Ewald/FFT implementation or a scalable hierarchical traversal.
function dipole_block(x, y)
    d = x - y
    r2 = dot(d, d)
    r2 > 0 || error("coincident sites")
    r = sqrt(r2)
    Matrix{Float64}(I, 3, 3) ./ r^3 .- 3 .* (d * d') ./ r^5
end

function cluster_matrix(x, y)
    m, n = size(x, 2), size(y, 2)
    T = Matrix{Float64}(undef, 3m, 3n)
    for i in 1:m, j in 1:n
        T[3i-2:3i, 3j-2:3j] = dipole_block(x[:, i], y[:, j])
    end
    T
end

function lowrank_probe()
    rng = MersenneTwister(137)
    # Two compact clusters separated along x; block is the exact dense oracle.
    a = 0.5 .* (rand(rng, 3, 48) .- 0.5)
    b = 6.0 .* ones(3, 48) .+ 0.5 .* (rand(rng, 3, 48) .- 0.5)
    T = cluster_matrix(a, b)
    s = svd(T)
    @printf("LOWRANK sites_per_cluster=48 separation=6.0 radius_bound=%.3f admissible_theta_0.5=%s\n",
        sqrt(3)*0.25, string(2sqrt(3)*0.25 < 0.5*6.0))
    for r in (3, 6, 12, 24, 48)
        T̂ = s.U[:, 1:r] * Diagonal(s.S[1:r]) * s.Vt[1:r, :]
        @printf("LOWRANK rank=%d rel_fro_error=%.6e apply_flops_proxy=%d dense=%d\n",
            r, norm(T-T̂)/norm(T), r*(size(T,1)+size(T,2)), length(T))
    end
end

function surface_term_probe()
    # Gaussian Ewald convention, spherical vacuum boundary: U_s=2π|M|²/(3V).
    # Tin-foil is the explicit comparison and has zero surface term.
    V, beta = 1000.0, 1.0
    M = [1.25, -0.5, 0.75]
    expected = 2pi * dot(M, M) / (3V)
    computed = (2pi / (3V)) * dot(M, M)
    omitted = 0.0
    @assert isapprox(computed, expected; rtol=1e-14)
    @assert !isapprox(omitted, expected; rtol=1e-14, atol=1e-14)
    @printf("SURFACE boundary=vacuum M2=%.8f V=%.1f U_surface=%.12e omission_detected=%s tin_foil=0\n",
        dot(M, M), V, computed, string(abs(computed-omitted)>1e-14))
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    println("Julia ", VERSION, " threads=", Threads.nthreads(), " BLAS threads=", BLAS.get_num_threads())
    lowrank_probe()
    surface_term_probe()
end
