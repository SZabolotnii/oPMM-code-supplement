import Std

namespace MvPATPMIMO

/-- Operational alpha range used by the implementation: 0.3 <= alpha <= 3.0.
We store tenths so the statement stays exact in Lean's rational-free core. -/
def AlphaTenth := Nat

def alphaInRange (a : AlphaTenth) : Prop :=
  3 <= a ∧ a <= 30

theorem alpha_three_admissible : alphaInRange 30 := by
  constructor <;> decide

theorem alpha_point_three_admissible : alphaInRange 3 := by
  constructor <;> decide

/-- A symbolic representation of the score family. -/
inductive ScoreKind where
  | linear
  | signedFractional (alphaTenth : AlphaTenth)
  | pmm3Cubic
  deriving DecidableEq, Repr

def opmmScore (a : AlphaTenth) : ScoreKind :=
  ScoreKind.signedFractional a

def pmm3Score : ScoreKind :=
  ScoreKind.pmm3Cubic

theorem opmm_score_is_signed_fractional (a : AlphaTenth) :
    opmmScore a = ScoreKind.signedFractional a := rfl

/-- Cumulant tuple matching the manuscript formula for `g(alpha; C)`. -/
structure CumulantVector where
  kappa2 : Int
  c1alpha : Int
  nualpha : Int
  sigma2alpha : Int
  mualpha : Int
  deriving DecidableEq, Repr

def gNumerator (C : CumulantVector) : Int :=
  C.kappa2 * C.sigma2alpha - C.c1alpha * C.c1alpha

def gDenominatorCore (alpha : Int) (C : CumulantVector) : Int :=
  C.sigma2alpha
    - 2 * alpha * C.nualpha * C.c1alpha
    + alpha * alpha * C.nualpha * C.nualpha * C.kappa2

def gDenominator (alpha : Int) (C : CumulantVector) : Int :=
  C.kappa2 * gDenominatorCore alpha C

theorem g_denominator_expands (alpha : Int) (C : CumulantVector) :
    gDenominator alpha C =
      C.kappa2 *
        (C.sigma2alpha
          - 2 * alpha * C.nualpha * C.c1alpha
          + alpha * alpha * C.nualpha * C.nualpha * C.kappa2) := rfl

end MvPATPMIMO
