module PiecewisePolynomials
# The approach here follows "Numerical Technique for the Convolution
# of Piecewise Polynomial Functions," Polge and Hays, 1972
# TODO: conversion to/from Matlab style mkpp() representation,
# spline interpolation, etc

using Polynomials

export DeltaIntegral, PiecewisePoly

import Base: convert, conv
import Polynomials: polyval, polyder, polyint

# order'th integral of the Dirac delta function
# TODO: check/warn/error if order <= 0
immutable DeltaIntegral{T<:Real, P<:Integer}
    shift::T
    order::P
    coef::T
end

function DeltaIntegral(shift::Real, order::Integer, coef::Real)
    T = promote_type(typeof(shift), typeof(coef))
    DeltaIntegral(convert(T, shift), order, convert(T, coef))
end

convert{T,P}(::Type{DeltaIntegral{T,P}}, p::DeltaIntegral) =
    DeltaIntegral(convert(T, p.shift), convert(P, p.order), convert(T, p.coef))

function polyval{T,P}(p::DeltaIntegral{T,P}, x::Real)
    ordf = float(p.order)
    R = promote_type(T, typeof(x), typeof(ordf))
    x < p.shift ? zero(R) :
        p.coef * convert(R, x-p.shift)^(p.order-1) / gamma(ordf)
    # TODO: compare performance vs range tradeoff of gamma vs factorial
end
polyval(p::DeltaIntegral, v::AbstractVector) = map(x->polyval(p, x), v)

+(p::DeltaIntegral) = p
-(p::DeltaIntegral) = DeltaIntegral(p.shift, p.order, -p.coef)
*(c::Real, p::DeltaIntegral) = DeltaIntegral(p.shift, p.order, c * p.coef)
*(p::DeltaIntegral, c::Real) = DeltaIntegral(p.shift, p.order, p.coef * c)
/(p::DeltaIntegral, c::Real) = DeltaIntegral(p.shift, p.order, p.coef / c)
polyder(p::DeltaIntegral) = DeltaIntegral(p.shift, p.order-1, p.coef)
polyint(p::DeltaIntegral) = DeltaIntegral(p.shift, p.order+1, p.coef)

conv(p1::DeltaIntegral, p2::DeltaIntegral) =
    DeltaIntegral(p1.shift + p2.shift, p1.order + p2.order, p1.coef * p2.coef)

# piecewise polynomial represented as a sum of delta integrals - the storage
# array of DeltaIntegral's is assumed to always be sorted by shift then order
immutable PiecewisePoly{T<:Real, P<:Integer}
    elems::Vector{DeltaIntegral{T,P}}
end

convert(::Type{PiecewisePoly}, p::DeltaIntegral) = PiecewisePoly([p])
+(p1::DeltaIntegral, p2::DeltaIntegral) = PiecewisePoly([p1]) + PiecewisePoly([p2])
-(p1::DeltaIntegral, p2::DeltaIntegral) = PiecewisePoly([p1]) - PiecewisePoly([p2])
+(p1::DeltaIntegral, p2::PiecewisePoly) = PiecewisePoly([p1]) + p2
-(p1::DeltaIntegral, p2::PiecewisePoly) = PiecewisePoly([p1]) - p2
+(p1::PiecewisePoly, p2::DeltaIntegral) = p1 + PiecewisePoly([p2])
-(p1::PiecewisePoly, p2::DeltaIntegral) = p1 - PiecewisePoly([p2])

polyval(pp::PiecewisePoly, x::Real) = sum([polyval(p, x) for p in pp.elems])
polyval(pp::PiecewisePoly, v::AbstractVector) = map(x->polyval(pp, x), v)

+(pp::PiecewisePoly) = pp
-(pp::PiecewisePoly) = PiecewisePoly([-p for p in pp.elems])
*(c::Real, pp::PiecewisePoly) = PiecewisePoly([c * p for p in pp.elems])
*(pp::PiecewisePoly, c::Real) = PiecewisePoly([p * c for p in pp.elems])
/(pp::PiecewisePoly, c::Real) = PiecewisePoly([p / c for p in pp.elems])
polyder(pp::PiecewisePoly) = PiecewisePoly([polyder(p) for p in pp.elems])
polyint(pp::PiecewisePoly) = PiecewisePoly([polyint(p) for p in pp.elems])

for op in (:+, :-)
    @eval begin
        function ($op){T1,T2,P1,P2}(pp1::PiecewisePoly{T1,P1}, pp2::PiecewisePoly{T2,P2})
            T = promote_type(T1, T2)
            P = promote_type(P1, P2)
            len1 = length(pp1.elems)
            len2 = length(pp2.elems)
            elems = Array(DeltaIntegral{T,P}, len1 + len2)
            i1, i2, j = 1, 1, 1
            # merge sorted arrays, combining elements with equal shift and order
            while i1 <= len1 && i2 <= len2
                p1 = pp1.elems[i1]
                p2 = pp2.elems[i2]
                if p1.shift < p2.shift
                    elems[j] = p1
                    j += 1
                    i1 += 1
                elseif p1.shift > p2.shift
                    elems[j] = ($op)(p2)
                    j += 1
                    i2 += 1
                elseif p1.shift == p2.shift
                    if p1.order < p2.order
                        elems[j] = p1
                        j += 1
                        i1 += 1
                    elseif p1.order > p2.order
                        elems[j] = ($op)(p2)
                        j += 1
                        i2 += 1
                    elseif p1.order == p2.order
                        coef = ($op)(p1.coef, p2.coef)
                        if coef != 0
                            elems[j] = DeltaIntegral(p1.shift, p1.order, coef)
                            j += 1
                        end
                        i1 += 1
                        i2 += 1
                    else
                        error("Unexpected condition: p1.order = $(p1.order)," *
                            " p2.order = $(p2.order)")
                    end
                else
                    error("Unexpected condition: p1.shift = $(p1.shift)," *
                        " p2.shift = $(p2.shift)")
                end
            end
            while i1 <= len1
                elems[j] = pp1.elems[i1]
                j += 1
                i1 += 1
            end
            while i2 <= len2
                elems[j] = ($op)(pp2.elems[i2])
                j += 1
                i2 += 1
            end
            return PiecewisePoly(deleteat!(elems, j:(len1 + len2)))
        end
    end
end

conv(p1::DeltaIntegral, p2::PiecewisePoly) =
    PiecewisePoly([conv(p1, p) for p in p2.elems])
conv(p1::PiecewisePoly, p2::DeltaIntegral) =
    PiecewisePoly([conv(p, p2) for p in p1.elems])
conv(p1::PiecewisePoly, p2::PiecewisePoly) =
    sum([conv(p1, p) for p in p2.elems])

end # module PiecewisePolynomials
