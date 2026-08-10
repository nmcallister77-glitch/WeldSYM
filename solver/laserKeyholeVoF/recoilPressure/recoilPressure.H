#ifndef recoilPressure_H
#define recoilPressure_H

#include "fvMesh.H"
#include "volFields.H"
#include "IOdictionary.H"

/*---------------------------------------------------------------------------*\
                    Class recoilPressureModel Declaration

    Knudsen-layer recoil pressure:
      P_r = prefactor * P_atm * exp(L_v * M * (T - T_b) / (R * T * T_b))
\*---------------------------------------------------------------------------*/

class recoilPressureModel
{
        const fvMesh& mesh_;
        const IOdictionary& thermoDict_;

        scalar prefactor_;
        scalar pAtm_;
        scalar Tb_;
        scalar Lv_;
        scalar M_;
        scalar Rgas_;
        scalar clampMin_;
        scalar clampMax_;

    public:

        recoilPressureModel(const fvMesh& mesh, const IOdictionary& thermo);

        void compute
        (
            const volScalarField& T,
            volScalarField& P_r
        ) const;

        //- Return as momentum source for interface cells
        volVectorField momentumSource
        (
            const volScalarField& P_r,
            const volScalarField& alphaMetal
        ) const;
};

#endif

// ************************************************************************* //
