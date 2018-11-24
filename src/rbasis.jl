using FEDVR
using BSplines
using FiniteDifferences
using IntervalSets

RBasis = Union{FEDVR.Basis,BSplines.Basis,FiniteDifferences.Basis}
for m in [:FEDVR, :BSplines, :FiniteDifferences]
    @eval locs(basis::$m.Basis) = $m.locs(basis)
end
extents(R::RBasis) = ..(extrema(locs(R))...)
