using HCubature
using ForwardDiff

function integrand(x, K) 
    #https://en.wikipedia.org/wiki/Ising_model
    log(cosh(2*K)^2 - sinh(2*K) *cos(x[1]) - sinh(2*K)*cos(x[2]))
end

function lnZ(K)
    log(2) + hcubature(x->integrand(x, K), (0.0, 0.0), (2*pi, 2*pi), rtol=1e-10)[1]/(8*pi^2) 
end 

dlnZ(K::Vector) = ForwardDiff.gradient(K->lnZ(K[1]), K)[1];
dlnZ2(K::Vector) = ForwardDiff.gradient(K->dlnZ(K), K)[1];

for K in collect(0.4:0.002:0.5)
    println(K, " ", lnZ(K), " ", -dlnZ([K]), " ", dlnZ2([K])*K^2)
end
