import Lake
open Lake DSL

package MvPATPMIMO where
  srcDir := "Lean"

lean_lib MvPATPMIMO where
  roots := #[`MvPATPMIMO]
