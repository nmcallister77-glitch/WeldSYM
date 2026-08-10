/*---------------------------------------------------------------------------*\
  laserKeyholeVoF — 3D keyhole-mode laser welding VOF solver (skeleton)

  Extends interFoam-style VOF with:
    - Enthalpy-porosity phase change (melting / solidification)
    - Knudsen-layer recoil pressure at vapor-liquid interface
    - Dual-cylinder Gaussian ray-tracing laser heat source
    - Metal vapor mass source / expansion
    - Marangoni + CSF surface tension + Boussinesq buoyancy

  Build:
    cd $WM_PROJECT_USER_DIR/applications/solvers/laserKeyholeVoF
    wmake

  Run:
    laserKeyholeVoF
    mpirun -np N laserKeyholeVoF -parallel
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "CMULES.H"
#include "EulerDdtScheme.H"
#include "localEulerDdtScheme.H"
#include "CrankNicolsonDdtScheme.H"
#include "subCycle.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "CorrectPhi.H"
#include "fvcSmooth.H"

#include "laserHeatSource.H"
#include "recoilPressure.H"
#include "rayTracer.H"
#include "enthalpyPorositySource.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Keyhole laser welding VOF solver with ray-tracing heat source,\n"
        "enthalpy-porosity phase change, and Knudsen recoil pressure."
    );

    #include "postProcess.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createAlphaFluxes.H"
    #include "initContinuityErrs.H"

    // Custom physics modules
    laserHeatSource laser(mesh);
    recoilPressureModel recoil(mesh, laserDict);
    rayTracer optics(mesh, laser);
    enthalpyPorositySource porous(mesh, thermoDict);

    turbulence->validate();

    if (!LTS)
    {
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "alphaCourantNo.H"
            #include "setDeltaT.H"
        }

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Adaptive mesh refinement (keyhole region) ---
        if (mesh.changing())
        {
            Info<< "Executing mesh update (AMR)" << endl;
            // alpha1, U, T mapped by dynamicRefineFvMesh
        }

        // --- Ray-tracing laser absorption (sub-cycled) ---
        for (label subCycle = 0; subCycle < laser.nSubCycles(); ++subCycle)
        {
            if (laser.model() == "dualCylinderGaussian")
            {
                optics.trace(laser, alpha1, laserHeatFlux, recoil);
            }
            else
            {
                laser.applyAbsorbedFlux(recoil, laserHeatFlux);
            }
        }

        // --- Phase change: update liquid fraction from enthalpy ---
        porous.updateLiquidFraction(liquidFraction, T);

        // --- Momentum predictor with recoil + Marangoni + porous drag ---
        while (pimple.loop())
        {
            if (pimple.firstPimpleIter() || mesh.changing())
            {
                // VOF alpha transport
                #include "alphaControls.H"
                #include "alphaEqnSubCycle.H"

                mixture.correct();

                // Update mixture density and mass flux after alpha changed
                rho = alpha1*mixture.rho1() + alpha2*mixture.rho2();
                rhoPhi = fvc::interpolate(rho)*phi;
            }

            if (pimple.freshRestart())
            {
                mixture.correct();
            }

            // Energy equation (enthalpy with latent heat + vaporization)
            #include "TEqn.H"

            // Momentum with source terms
            #include "UEqn.H"

            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
        }

        // Vapor mass source at boiling interface
        porous.applyVaporizationSource(alpha2, alpha1, T);

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;
    return 0;
}

// ************************************************************************* //
