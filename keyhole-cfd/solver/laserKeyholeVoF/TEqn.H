{
    // Temperature equation: rho*Cp*(dT/dt + U.Grad T) = Div(kEff Grad T) + Q_laser + ...
    // laserHeatFlux is a cell-centred volumetric source [W/m^3].

    rhoCp = rho*Cp;
    alpha = kappa/(rhoCp);
    kEff = rhoCp*alpha;
    rhoCpPhi = rhoPhi*Cp;

    fvScalarMatrix TEqn
    (
        fvm::ddt(rhoCp, T)
      + fvm::div(rhoCpPhi, T)
      - fvm::laplacian(kEff, T)
     ==
        laserHeatFlux
      + fvOptions(rhoCp, T)
    );

    TEqn.relax();
    fvOptions.constrain(TEqn);
    TEqn.solve();
    fvOptions.correct(T);

    porous.updateLiquidFraction(liquidFraction, T);
    recoil.compute(T, recoilPressure);
}
