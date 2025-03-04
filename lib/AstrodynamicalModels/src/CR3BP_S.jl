
using Symbolics
using SciMLBase
using Memoize
using LinearAlgebra
using ModelingToolkit, OrdinaryDiffEq
using StaticArrays

__revise_mode__ = :eval


@memoize function CR3BSystem(;
    stm = false,
    name = :CR3B,
    defaults = Pair{ModelingToolkit.Num,<:Number}[],
    kwargs...,
)
    @independent_variables t
    @parameters μ α δ β
    @variables x(t) y(t) z(t) ẋ(t) ẏ(t) ż(t)
    delta = Differential(t) 
    r = [x, y, z]
    v = [ẋ, ẏ, ż]
    n = [cosd(α)*x/norm(r) - cosd(α)sind(δ)*x*z/(norm(r)*norm([x, y])) - sind(α)*y/norm(r) + sind(α)*sind(δ)*y*z/(norm(r)*norm([x, y])), 
         cosd(α)*y/norm(r) - cosd(α)sind(δ)*y*z/(norm(r)*norm([x, y])) + sind(α)*x/norm(r) - sind(α)*sind(δ)*x*z/(norm(r)*norm([x, y])), 
         z/norm(r) + sind(δ)*norm([x, y])/norm(r)]
    eqs = vcat(
        delta.(r) .~ v,
        delta(ẋ) ~
            (x + 2ẏ - (μ * (x + μ - 1) * (sqrt(y^2 + z^2 + (x + μ - 1)^2)^-3)) -
            ((x + μ) * (sqrt(y^2 + z^2 + (x + μ)^2)^-3) * (1 - μ))) + β * (1-μ)/norm(r)^2 * dot(r/norm(r), n)^2 * n[1]
            ,
        delta(ẏ) ~
            y - (2ẋ) - (
                y * (
                    μ * (sqrt(y^2 + z^2 + (x + μ - 1)^2)^-3) +
                    (sqrt(y^2 + z^2 + (x + μ)^2)^-3) * (1 - μ)
                )
            ) + β * (1-μ)/norm(r)^2 * dot(r/norm(r), n)^2 * n[2],
        delta(ż) ~
            z * (
                -μ * (sqrt(y^2 + z^2 + (x + μ - 1)^2)^-3) -
                ((sqrt(y^2 + z^2 + (x + μ)^2)^-3) * (1 - μ))
            ) + β * (1-μ)/norm(r)^2 * dot(r/norm(r), n)^2 * n[3],
    )

    if stm
        @variables (Φ(t))[1:6, 1:6] [description = "state transition matrix estimate"]
        A = Symbolics.jacobian(map(el -> el.rhs, eqs), vcat(r, v))

        Φ = Symbolics.scalarize(Φ)
        LHS = delta.(Φ)
        RHS = A * Φ

        eqs = vcat(eqs, vec([LHS[i] ~ RHS[i] for i in eachindex(LHS)]))
    end

    if string(name) == "CR3B" && stm
        modelname = Symbol("CR3BWithSTM")
    else
        modelname = name
    end

    if stm
        defaults = vcat(defaults, vec(Φ .=> Float64.(I(6))))
        return ODESystem(
            eqs,
            t,
            vcat(r, v, vec(Φ)),
            [μ, α, δ, β];
            name = modelname,
            defaults = defaults,
            kwargs...,
        )
    else
        return ODESystem(
            eqs,
            t,
            vcat(r, v),
            [μ, α, δ, β];
            name = modelname,
            defaults = defaults,
            kwargs...,
        )
    end
end

@memoize function CR3B_SFunction(; stm = false, name = :CR3B, kwargs...)
    defaults = (; jac = true)
    options = merge(defaults, kwargs)
    sys = complete(CR3BSystem(; stm = stm, name = name); split = false)
    return ODEFunction{true,SciMLBase.FullSpecialize}(
        sys,
        ModelingToolkit.unknowns(sys),
        ModelingToolkit.parameters(sys);
        options...,
    )
end

CR3B_SProblem(u0, tspan, p; kwargs...) = ODEProblem(CR3B_SFunction(), u0, tspan, p; kwargs...)

"""
Solve for the monodromy matrix of the periodic orbit.
"""
function monodromy_S(
    u::AbstractVector,
    μ,
    α,
    δ,
    β,
    T,
    f;
    algorithm = Vern9(),
    reltol = 1e-15,
    abstol = 1e-15,
    save_everystep = false,
    kwargs...,
)
    problem = ODEProblem(
        f,
        MVector{42}(
            u[begin],
            u[begin+1],
            u[begin+2],
            u[begin+3],
            u[begin+4],
            u[begin+5],
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
        ),
        (zero(T), T),
        (μ, α, δ, β),
    )
    solution = solve(
        problem,
        algorithm;
        reltol = reltol,
        abstol = abstol,
        save_everystep = save_everystep,
        kwargs...,
    )

    if solution.u[begin][begin:begin+5] ≉ solution.u[end][begin:begin+5]
        @warn "The orbit does not appear to be periodic!"
    end

    return reshape((solution.u[end][begin+6:end]), 6, 6)
end
