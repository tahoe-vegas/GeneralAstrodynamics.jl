#
# Circular Restricted Three-body Problem with Solar Sail
#


Base.@kwdef struct CR3B_SailParameters{F} <: AstrodynamicalParameters{F,4}
    μ::F
    α::F
    δ::F
    β::F

    CR3B_SailParameters{F}(μ::Number, α::Number, δ::Number, β::Number) where {F} = new{F}(μ, α, δ, β)
    CR3B_SailParameters(μ::Number, α::Number, δ::Number, β::Number) = new{typeof(μ)}(μ, α, δ, β)
    CR3B_SailParameters{F}(μ::Tuple) where {F} = CR3B_SailParameters{F}(μ...)
    CR3B_SailParameters(μ::Tuple) = CR3B_SailParameters(μ...)

end

system(::CR3B_SailParameters, args...; kwargs...) = CR3B_SailSystem(args...; kwargs...)
dynamics(::CR3B_SailParameters, args...; kwargs...) = CR3B_SailFunction(args...; kwargs...)
Base.@pure paradigm(::CR3B_SailParameters) = "Circular Restricted Three Body Solar Sail Dynamics"


@memoize function CR3B_SailSystem(;
    stm = false,
    name = :CR3B_Sail,
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

    if string(name) == "CR3B_Sail" && stm
        modelname = Symbol("CR3B_SailWithSTM")
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

@memoize function CR3B_SailFunction(; stm = false, name = :CR3B, kwargs...)
    defaults = (; jac = true)
    options = merge(defaults, kwargs)
    sys = complete(CR3B_SailSystem(; stm = stm, name = name); split = false)
    return ODEFunction{true,SciMLBase.FullSpecialize}(
        sys,
        ModelingToolkit.unknowns(sys),
        ModelingToolkit.parameters(sys);
        options...,
    )
end

const CR3B_SailOrbit = Orbit{<:CR3BState,<:CR3B_SailParameters}
AstrodynamicalModels.CR3B_SailOrbit(state::AbstractVector, parameters::AbstractVector) =
    Orbit(CR3BState(state), CR3B_SailParameters(parameters))
AstrodynamicalModels.CR3B_SailOrbit(; state::AbstractVector, parameters::AbstractVector) =
    Orbit(CR3BState(state), CR3B_SailParameters(parameters))


CR3B_SailProblem(u0, tspan, p; kwargs...) = ODEProblem(CR3B_SailFunction(), u0, tspan, p; kwargs...)
