{
    // Temperature equation: rho*Cp*(dT/dt + U.Grad T) = Div(kEff Grad T) + Q_laser + ...
    // laserHeatFlux is a cell-centred volumetric source [W/m^3].

    alpha = kappa/(rho*Cp);
    kEff = rho*Cp*alpha;
    rhoCpPhi = fvc::interpolate(rho*Cp)*phi;

    fvScalarMatrix TEqn
    (
        Cp*fvm::ddt(rho, T)
      + fvm::div(rhoCpPhi, T)
      - fvm::laplacian(kEff, T)
     ==
        laserHeatFlux
      + fvOptions(T)
    );

    TEqn.relax();
    fvOptions.constrain(TEqn);
    TEqn.solve();
    fvOptions.correct(T);

    porous.updateLiquidFraction(liquidFraction, T);
    recoil.compute(T, recoilPressure);
}
