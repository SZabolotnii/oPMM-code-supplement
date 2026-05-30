import Std

namespace MvPATPMIMO

/-- Trace normalization used for matrix row covariance: tr(Gamma) = k. -/
structure TraceState where
  dimension : Nat
  traceTenth : Nat
  deriving DecidableEq, Repr

def traceNormalized (s : TraceState) : Prop :=
  s.traceTenth = 10 * s.dimension

def normalizeTrace (s : TraceState) : TraceState :=
  { s with traceTenth := 10 * s.dimension }

theorem normalize_trace_normalized (s : TraceState) :
    traceNormalized (normalizeTrace s) := by
  unfold traceNormalized normalizeTrace
  rfl

theorem normalized_trace_unique (s t : TraceState)
    (hdim : s.dimension = t.dimension)
    (hs : traceNormalized s)
    (ht : traceNormalized t) :
    s.traceTenth = t.traceTenth := by
  unfold traceNormalized at hs ht
  rw [hs, ht, hdim]

end MvPATPMIMO
