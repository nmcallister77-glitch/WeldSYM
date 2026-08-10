#ifndef laserHeatSource_H
#define laserHeatSource_H

#include "fvMesh.H"
#include "volFields.H"
#include "IOdictionary.H"

/*---------------------------------------------------------------------------*\
                    Class laserHeatSource Declaration
\*---------------------------------------------------------------------------*/

class laserHeatSource
{
    // Private Data
        const fvMesh& mesh_;
        IOdictionary dict_;
        word model_;
        scalar power_;
        scalar w0_;
        scalar sigmaZ_;
        vector focus_;
        vector direction_;
        vector travelDir_;
        scalar travelSpeed_;
        scalar absorptivity_;
        label nSubCycles_;

    public:

    // Constructors
        laserHeatSource(const fvMesh& mesh);

        static autoPtr<laserHeatSource> New(const fvMesh& mesh);

    // Member Functions
        const fvMesh& mesh() const { return mesh_; }
        const IOdictionary& thermo() const { return dict_; }

        const word& model() const { return model_; }
        scalar power() const { return power_; }
        scalar w0() const { return w0_; }
        const vector& direction() const { return direction_; }
        const vector& focus() const { return focus_; }
        scalar travelSpeed() const { return travelSpeed_; }
        scalar absorptivity() const { return absorptivity_; }
        label nSubCycles() const { return nSubCycles_; }

        //- Current laser focus position (moving heat source)
        vector currentFocus(scalar t) const;

        //- Gaussian beam intensity at radial distance r from axis [W/m²]
        scalar beamIntensity(scalar r, scalar z) const;

        //- Apply absorbed flux field from ray tracing
        void applyAbsorbedFlux
        (
            const recoilPressureModel& recoil,
            volScalarField& laserHeatFlux
        ) const;

        void applyAbsorbedFlux(const recoilPressureModel& recoil) const;
};

#endif

// ************************************************************************* //
