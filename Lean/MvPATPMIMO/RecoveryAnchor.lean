import MvPATPMIMO.Score

namespace MvPATPMIMO

/-- The recovery anchor is symbolic: at alpha = 3, the manuscript identifies
the oPMM score with the symmetric PMM3 cubic anchor. -/
def recoveryAnchor (a : AlphaTenth) : ScoreKind :=
  if a = 30 then pmm3Score else opmmScore a

theorem recovery_at_alpha_three :
    recoveryAnchor 30 = pmm3Score := by
  simp [recoveryAnchor]

theorem recovery_off_anchor (a : AlphaTenth) (h : a ≠ 30) :
    recoveryAnchor a = opmmScore a := by
  simp [recoveryAnchor, h]

theorem recovery_anchor_is_admissible :
    alphaInRange 30 := alpha_three_admissible

end MvPATPMIMO
