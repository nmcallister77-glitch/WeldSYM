// VOF alpha transport — delegates to MULES (see interFoam alphaEqnSubCycle.H)
// Copy full implementation from $FOAM_SOLVERS/multiphase/interFoam/alphaEqnSubCycle.H
// when integrating into OpenFOAM tree.

#include "alphaEqn.H"

mixture.correct();
