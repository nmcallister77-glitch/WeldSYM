#include "laserHeatSource.H"
#include "recoilPressure.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

laserHeatSource::laserHeatSource(const fvMesh& mesh)
:
    mesh_(mesh),
    dict_
    (
        IOobject
        (
            "laserHeatSource",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    ),
    model_(dict_.getOrDefault<word>("laserModel", "simpleGaussian")),
    power_(dict_.subDict("beam").get<scalar>("power")),
    w0_(dict_.subDict("beam").get<scalar>("w0")),
    sigmaZ_(dict_.subDict("beam").getOrDefault<scalar>("sigmaZ", w0_)),
    focus_(dict_.subDict("beam").get<vector>("focus")),
    direction_(dict_.subDict("beam").getOrDefault<vector>("direction", vector(0, 0, -1))),
    travelDir_(dict_.subDict("beam").get<vector>("travelDir")),
    travelSpeed_(dict_.subDict("beam").get<scalar>("travelSpeed")),
    absorptivity_(dict_.subDict("beam").get<scalar>("absorptivity")),
    nSubCycles_(dict_.subDict("rayTracing").get<label>("subCycles"))
{}


Foam::autoPtr<laserHeatSource> laserHeatSource::New(const fvMesh& mesh)
{
    return autoPtr<laserHeatSource>(new laserHeatSource(mesh));
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

vector laserHeatSource::currentFocus(scalar t) const
{
    return focus_ + travelDir_*travelSpeed_*t;
}


scalar laserHeatSource::beamIntensity(scalar r, scalar z) const
{
    // Gaussian: I(r) = (2P/(pi*w²)) * exp(-2r²/w²)
    const scalar w = w0_;  // TODO: add Rayleigh range w(z)
    const scalar I0 = 2.0*power_/(constant::mathematical::pi*sqr(w));
    return absorptivity_*I0*Foam::exp(-2.0*sqr(r)/sqr(w));
}


void laserHeatSource::applyAbsorbedFlux
(
    const recoilPressureModel& recoil,
    volScalarField& laserHeatFlux
) const
{
    (void)recoil;

    // Simple/fallback volumetric Gaussian heat source.
    // For dualCylinderGaussian the ray-tracer is expected to populate the field;
    // if it leaves it near-zero, this adds a moving 3D Gaussian source.
    const scalar t = mesh_.time().value();
    const vector fc = currentFocus(t);
    const vector dir = direction_/mag(direction_);
    const scalar P = power_*absorptivity_;

    // Radial and axial spread.  w0 is 1/e^2 radius; use 2D Gaussian normalisation.
    const scalar sigma = w0_;
    const scalar sigmaZ = sigmaZ_;
    const scalar twoSigma2 = 2.0*sqr(sigma);
    const scalar twoSigmaZ2 = 2.0*sqr(sigmaZ);

    // Normalisation so total integrated power = P (approximate for 3D Gaussian).
    const scalar norm = P
        / (pow(2.0*constant::mathematical::pi, 1.5)*sigma*sigma*sigmaZ);

    forAll(laserHeatFlux, cellI)
    {
        const vector r = mesh_.C()[cellI] - fc;
        const scalar z = r & dir;
        const vector rPerp = r - z*dir;
        const scalar r2 = magSqr(rPerp);
        const scalar z2 = sqr(z);

        laserHeatFlux[cellI] = norm*Foam::exp(-r2/twoSigma2)*Foam::exp(-z2/twoSigmaZ2);
    }

    Info<< "laserHeatSource: applied Gaussian volumetric source at focus " << fc << endl;
}


void laserHeatSource::applyAbsorbedFlux(const recoilPressureModel& recoil) const
{
    (void)recoil;
    WarningInFunction
        << "applyAbsorbedFlux() without a field is deprecated; "
        << "use applyAbsorbedFlux(recoil, laserHeatFlux)." << endl;
}

// ************************************************************************* //
