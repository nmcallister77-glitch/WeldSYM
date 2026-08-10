{
    volVectorField porousDrag(porous.momentumDamping(U, liquidFraction));
    volVectorField recoilSource(recoil.momentumSource(recoilPressure, alpha1));

    fvVectorMatrix UEqn
    (
        fvm::ddt(rho, U)
      + fvm::div(rhoPhi, U)
      + MRF.DDt(rho, U)
      + turbulence->divDevRhoReff(U)
     ==
        fvOptions(rho, U)
      + recoilSource
      + porousDrag
    );

    UEqn.relax();
    fvOptions.constrain(UEqn);

    if (pimple.momentumPredictor())
    {
        solve(UEqn == -fvc::grad(p_rgh) - gh*fvc::grad(rho));
    }
}
