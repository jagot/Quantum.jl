using WignerSymbols

function C(k, ℓ, ℓ′, q=0, m=0, m′=0)
    s = (-1)^(2ℓ-m)
    N = √((2ℓ+1)*(2ℓ′+1))
    W = wigner3j(ℓ, k, ℓ′,
                 -m, q, m′)
    Wr = wigner3j(ℓ, k, ℓ′,
                  0, 0, 0)
    s*N*W*Wr
end
