    MRF.correctBoundaryVelocity(U);

    volScalarField porousCoeff(porous.momentumDampingCoeff(liquidFraction));
    porousCoeff *= pos0(alpha1 - 0.5);
    volVectorField recoilSource(recoil.momentumSource(recoilPressure, alpha1));

    fvVectorMatrix UEqn
    (
        fvm::ddt(rho, U) + fvm::div(rhoPhi, U)
      + MRF.DDt(rho, U)
      + turbulence->divDevRhoReff(rho, U)
      + fvm::Sp(porousCoeff, U)
     ==
        fvOptions(rho, U)
      + recoilSource
    );

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (pimple.momentumPredictor())
    {
        solve
        (
            UEqn
         ==
            fvc::reconstruct
            (
                (
                    mixture.surfaceTensionForce()
                  - ghf*fvc::snGrad(rho)
                  - fvc::snGrad(p_rgh)
                ) * mesh.magSf()
            )
        );

        fvOptions.correct(U);
    }
