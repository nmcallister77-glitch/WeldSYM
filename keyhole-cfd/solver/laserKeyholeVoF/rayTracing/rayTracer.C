#include "fvCFD.H"
#include "rayTracer.H"
#include "Random.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

rayTracer::rayTracer(const fvMesh& mesh, const laserHeatSource& laser)
:
    mesh_(mesh)
{
    const IOdictionary& dict = laser.thermo();
    const dictionary& rt = dict.subDict("rayTracing");
    nRays_ = rt.get<label>("numRays");
    maxBounces_ = rt.get<label>("maxBounces");
    powerThreshold_ = rt.get<scalar>("powerThreshold");
    diffuseFraction_ = rt.get<scalar>("diffuseFraction");
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

scalar rayTracer::fresnelReflectance
(
    scalar n1,
    scalar n2,
    scalar thetaI
) const
{
    // Unpolarized Fresnel — stub uses fixed molten-metal n
    const scalar sinT = n1/n2 * Foam::sin(thetaI);
    if (mag(sinT) >= 1.0) return 1.0;

    const scalar cosI = Foam::cos(thetaI);
    const scalar cosT = Foam::sqrt(1.0 - sqr(sinT));

    const scalar rs = sqr((n1*cosI - n2*cosT)/(n1*cosI + n2*cosT));
    const scalar rp = sqr((n1*cosT - n2*cosI)/(n1*cosT + n2*cosI));
    return 0.5*(rs + rp);
}


bool rayTracer::intersectIsosurface
(
    const point& origin,
    const vector& dir,
    point& hit,
    vector& normal
) const
{
    // TODO: walk mesh octree / cut-cell isosurface of alpha.metal = 0.5
    (void)origin; (void)dir; hit = origin; normal = vector(0, 0, 1);
    return false;
}


void rayTracer::trace
(
    const laserHeatSource& laser,
    const volScalarField& alphaMetal,
    volScalarField& laserHeatFlux,
    recoilPressureModel& recoil
)
{
    (void)recoil;

    Info<< "rayTracer: tracing " << nRays_ << " rays (max "
        << maxBounces_ << " bounces)" << endl;

    // Fallback until full isosurface ray-tracing is implemented:
    // deposit laser power into metal cells near the beam axis.
    laserHeatFlux = dimensionedScalar(laserHeatFlux.dimensions(), Zero);

    const scalar t = mesh_.time().value();
    const point focus = laser.currentFocus(t);
    const vector dir = vector(0, 0, -1); // downward for now; should come from laser
    const scalar P = laser.power()*laser.absorptivity();
    const scalar sigma = laser.w0();
    const scalar twoSigma2 = 2.0*sqr(sigma);

    // Integrate beam footprint to get normalisation factor
    scalar totalWeight = 0.0;
    forAll(laserHeatFlux, cellI)
    {
        const point& cc = mesh_.C()[cellI];
        const scalar zProj = (cc - focus) & dir;
        const vector rPerp = (cc - focus) - zProj*dir;
        const scalar r2 = magSqr(rPerp);
        const scalar w = Foam::exp(-r2/twoSigma2) * (alphaMetal[cellI] > 0.01 ? 1.0 : 0.0);
        totalWeight += w;
    }

    reduce(totalWeight, sumOp<scalar>());

    if (totalWeight > ROOTVSMALL)
    {
        const scalar qPerWeight = P / (totalWeight * mesh_.V()[0]); // rough per-cell

        forAll(laserHeatFlux, cellI)
        {
            const point& cc = mesh_.C()[cellI];
            const scalar zProj = (cc - focus) & dir;
            const vector rPerp = (cc - focus) - zProj*dir;
            const scalar r2 = magSqr(rPerp);
            const scalar w = Foam::exp(-r2/twoSigma2) * (alphaMetal[cellI] > 0.01 ? 1.0 : 0.0);
            // Distribute power weighted by volume to keep units W/m^3
            laserHeatFlux[cellI] = P * w / (totalWeight * mesh_.V()[cellI]);
        }
    }

    Info<< "rayTracer: deposited " << P << " W into "
        << (totalWeight > ROOTVSMALL ? "metal field" : "(no metal found)") << endl;
}

// ************************************************************************* //
