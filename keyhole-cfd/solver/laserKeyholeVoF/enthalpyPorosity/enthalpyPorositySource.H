#ifndef enthalpyPorositySource_H
#define enthalpyPorositySource_H

#include "fvMesh.H"
#include "volFields.H"
#include "IOdictionary.H"

/*---------------------------------------------------------------------------*\
                  Class enthalpyPorositySource Declaration

  Enthalpy-porosity mushy zone momentum damping + liquid fraction update +
  vaporization mass source at T >= Tb.
\*---------------------------------------------------------------------------*/

class enthalpyPorositySource
{
        const fvMesh& mesh_;
        scalar beta_;
        scalar epsilon_;
        scalar vaporCoeff_;
        scalar Ts_;
        scalar Tl_;
        scalar Tb_;
        scalar Lf_;
        scalar Lv_;

    public:

        enthalpyPorositySource(const fvMesh& mesh, const IOdictionary& thermo);

        void updateLiquidFraction
        (
            volScalarField& liquidFraction,
            const volScalarField& T
        ) const;

        void updateLiquidFraction();

        volVectorField momentumDamping
        (
            const volVectorField& U,
            const volScalarField& liquidFraction
        ) const;

        volScalarField momentumDampingCoeff
        (
            const volScalarField& liquidFraction
        ) const;

        void applyVaporizationSource
        (
            volScalarField& alphaVapor,
            const volScalarField& alphaMetal,
            const volScalarField& T
        ) const;

        void applyVaporizationSource();
};

#endif

// ************************************************************************* //
