// Simple bounded explicit alpha transport — placeholder until the full
// interFoam MULES alpha equation is copied into this solver.
{
    // Read controls from fvSolution under "alpha.metal*"
    const dictionary& alphaControls = mesh.solverDict(alpha1.name());
    const label nAlphaCorr = alphaControls.lookupOrDefault<label>("nAlphaCorr", 1);

    for (label aCorr = 0; aCorr < nAlphaCorr; ++aCorr)
    {
        // First-order upwind face flux for the volume fraction
        surfaceScalarField phiAlpha
        (
            "phiAlpha",
            fvc::interpolate(alpha1)*phi
        );

        // Explicit Euler: alpha^(n+1) = alpha^n - dt * div(phiAlpha)
        // MULES/isoAdvection should replace this in production.
        alpha1 = alpha1 - runTime.deltaT()*fvc::div(phiAlpha);

        // Brute-force bounding [0,1] — not conservative, keeps field sane.
        forAll(alpha1, cellI)
        {
            alpha1[cellI] = Foam::max(Foam::min(alpha1[cellI], 1.0), 0.0);
        }
    }

    alpha2 = 1.0 - alpha1;
}
