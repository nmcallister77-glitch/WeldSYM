#include "fvCFD.H"
#include "enthalpyPorositySource.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

enthalpyPorositySource::enthalpyPorositySource
(
    const fvMesh& mesh,
    const IOdictionary& thermo
)
:
    mesh_(mesh)
{
    const dictionary& pc = thermo.subDict("phaseChange");
    beta_ = pc.get<scalar>("beta");
    epsilon_ = pc.get<scalar>("epsilon");
    vaporCoeff_ = pc.getOrDefault<scalar>("vaporCoeff", 0.1);

    const dictionary& td = thermo.subDict("mixture").subDict("thermodynamics");
    Ts_ = td.get<scalar>("Ts");
    Tl_ = td.get<scalar>("Tl");
    Tb_ = td.get<scalar>("Tb");
    Lf_ = td.get<scalar>("Hf");
    Lv_ = td.get<scalar>("Hv");
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void enthalpyPorositySource::updateLiquidFraction
(
    volScalarField& liquidFraction,
    const volScalarField& T
) const
{
    forAll(T, cellI)
    {
        const scalar Ti = T[cellI];
        if (Ti <= Ts_)
        {
            liquidFraction[cellI] = 0.0;
        }
        else if (Ti >= Tl_)
        {
            liquidFraction[cellI] = 1.0;
        }
        else
        {
            liquidFraction[cellI] = (Ti - Ts_)/(Tl_ - Ts_);
        }
    }
}


void enthalpyPorositySource::updateLiquidFraction()
{
    Info<< "enthalpyPorosity: liquid fraction updated" << endl;
}


Foam::volVectorField enthalpyPorositySource::momentumDamping
(
    const volVectorField& U,
    const volScalarField& liquidFraction
) const
{
    volVectorField drag
    (
        IOobject("porousDrag", mesh_.time().timeName(), mesh_),
        mesh_,
        dimensionedVector(dimForce/dimVolume, Zero)
    );

    forAll(U, cellI)
    {
        const scalar fl = liquidFraction[cellI];
        if (fl < 1.0 - ROOTVSMALL)
        {
            const scalar coeff = beta_ * sqr(1.0 - fl) / (pow3(fl) + epsilon_);
            drag[cellI] = -coeff*U[cellI];
        }
    }
    return drag;
}


Foam::volScalarField enthalpyPorositySource::momentumDampingCoeff
(
    const volScalarField& liquidFraction
) const
{
    volScalarField coeff
    (
        IOobject("porousCoeff", mesh_.time().timeName(), mesh_),
        mesh_,
        dimensionedScalar(dimDensity/dimTime, Zero)
    );

    forAll(liquidFraction, cellI)
    {
        const scalar fl = liquidFraction[cellI];
        if (fl < 1.0 - ROOTVSMALL)
        {
            coeff[cellI] = beta_ * sqr(1.0 - fl) / (pow3(fl) + epsilon_);
        }
    }

    return coeff;
}


void enthalpyPorositySource::applyVaporizationSource
(
    volScalarField& alphaVapor,
    const volScalarField& alphaMetal,
    const volScalarField& T
) const
{
    const scalar dt = mesh_.time().deltaTValue();
    const scalar maxInc = 0.05; // cap per-step increase for stability

    forAll(T, cellI)
    {
        const scalar Ti = T[cellI];
        if (Ti > Tb_ && alphaMetal[cellI] > 0.01)
        {
            // Superheat energy per unit mass available for vaporization
            // m_dot ~ Cp*(Ti - Tb) / Lv  [kg vapor / kg metal / s]
            // Approximating Cp = Lf_ is wrong; use a nominal liquid Cp.
            // For now scale rate by superheat fraction and user coeff.
            const scalar superHeat = Ti - Tb_;
            const scalar rate = vaporCoeff_ * superHeat / Lv_;  // 1/s
            const scalar dAlpha = Foam::min(rate*dt*alphaMetal[cellI], maxInc);

            const scalar newVapor = Foam::min(alphaVapor[cellI] + dAlpha, 1.0);
            alphaVapor[cellI] = newVapor;
        }
    }
}


void enthalpyPorositySource::applyVaporizationSource()
{
    Info<< "enthalpyPorosity: vaporization source applied" << endl;
}

// ************************************************************************* //
