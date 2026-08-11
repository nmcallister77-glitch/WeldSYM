#ifndef rayTracer_H
#define rayTracer_H

#include "fvMesh.H"
#include "volFields.H"
#include "laserHeatSource.H"
#include "recoilPressure.H"

/*---------------------------------------------------------------------------*\
                         Class rayTracer Declaration

  Dual-cylinder Gaussian multi-bounce Fresnel ray tracing against alpha=0.5
  isosurface. Populates laserHeatFlux and satisfies interface energy balance.
\*---------------------------------------------------------------------------*/

class rayTracer
{
        const fvMesh& mesh_;
        label nRays_;
        label maxBounces_;
        scalar powerThreshold_;
        scalar diffuseFraction_;

    public:

        rayTracer(const fvMesh& mesh, const laserHeatSource& laser);

        void trace
        (
            const laserHeatSource& laser,
            const volScalarField& alphaMetal,
            volScalarField& laserHeatFlux,
            recoilPressureModel& recoil
        );

    private:

        scalar fresnelReflectance(scalar n1, scalar n2, scalar thetaI) const;
        bool intersectIsosurface
        (
            const point& origin,
            const vector& dir,
            point& hit,
            vector& normal
        ) const;
};

#endif

// ************************************************************************* //
