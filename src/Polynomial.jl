    # Poly type manipulations

module Polynomial

export Poly, polyval, polyint, polydir, poly, roots, papply, integrate, derivative

import Base.length, Base.endof, Base.getindex, Base.setindex!, Base.copy, Base.zero, Base.one, Base.promote_rule
import Base.show, Base.*, Base./, Base.-, Base.+, Base.==

type Poly{T<:Number}
    a::Vector{T}
    function Poly(a::Vector{T})
        new(a)
    end
end

immutable PolynomialUnit{s}; end

Poly{T<:Number}(a::Vector{T}) = Poly{T}(a)

promote_rule{T}(::Type{T},::Type{Poly{T}}) = Poly{T}

length(p::Poly) = length(p.a)-1
endof(p::Poly) = length(p)
getindex{T}(p::Poly{T}, i) = (i+1 > length(p.a) ? zero(T) : p.a[i+1])
function setindex!(p::Poly, v, i) 
    n = length(p.a)
    if n < i+1
        resize!(p.a,i+1)
        p.a[n:i] = 0
    end
    p.a[i+1] = v
    v
end

copy(p::Poly) = Poly(copy(p.a))

zero{T}(p::Poly{T}) = zero(Poly{T})
zero{T}(::Type{Poly{T}}) = Poly(T[])
one{T}(p::Poly{T}) = one(Poly{T})
one{T}(::Type{Poly{T}}) = Poly([one(T)])

function print_exponent{T}(io::IO,p::Poly{T},i,first)
    if p[i] == zero(T)
        return false
    end
    if first != true
        print(io," + ")
    end
    if p[i] != one(T) || i == 0
        show(io,p[i])
    end
    if i == 0
    elseif i == 1
        print(io,"x")
    else
        print(io,"x^",i)
    end
    true
end

function show{T}(io::IO,p::Poly{T})
    n = length(p)
    print(io,"Poly(")
    first = true
    printed_anything = false
    for i = 0:n
        printed = print_exponent(io,p,i,first)
        first &= !printed
        printed_anything |= printed
    end
    printed_anything || print(io,zero(T))
    print(io,")")
end

*{T<:Number,S}(c::T, p::Poly{S}) = Poly(c * p.a)
*{T<:Number,S}(p::Poly{S}, c::T) = Poly(p.a * c)
/(p::Poly, c::Number) = Poly(p.a / c)
-(p::Poly) = Poly(-p.a)

-(p::Poly, c::Number) = +(p, -c)
+(c::Number, p::Poly) = +(p, c)
function +(p::Poly, c::Number)
    if length(p) < 1
        return Poly([c,])
    else
        p2 = copy(p)
        p2.a[1] += c;
        return p2;
    end
end
function -{T}(c::Number, p::Poly{T})
    if length(p) < 1
        return Poly(T[c,])
    else
        p2 = -p;
        p2.a[1] += c;
        return p2;
    end
end

+{T,S}(p1::Poly{T}, p2::Poly{S}) = Poly([p1[i] + p2[i] for i = 0:max(length(p1),length(p2))])
-{T,S}(p1::Poly{T}, p2::Poly{S}) = Poly([p1[i] - p2[i] for i = 0:max(length(p1),length(p2))])

function *{T,S}(p1::Poly{T}, p2::Poly{S})
    R = promote_type(T,S)
    n = length(p1)
    m = length(p2)
    a = Poly(zeros(R,m+n+2))
    for i = 0:length(p1)
        for j = 0:length(p2)
            a[i+j] += p1[i] * p2[j]
        end
    end
    a
end


function ==(p1::Poly, p2::Poly)
    if
        for i = 1:max(length(p1),length(p2))
            if p1[i] != p2_2[i]
                return false    
            end
        end
    end
    return true
end

function papply{T,S}(p1::Poly{T},x::S)
    ret = zero(x)
    for i = 0:length(p1)
        ret += p1[i] .* x^i
    end
    ret
end

integrate(p::Poly) = Poly([0;[1/(n+1)*p[n] for n=0:length(p)]])
derivative(p::Poly) = Poly([ n*p[n] for n=1:length(p) ])
derivative(vec::Vector) = map(derivative,vec)
derivative(x::Number) = 0

end # module Poly
