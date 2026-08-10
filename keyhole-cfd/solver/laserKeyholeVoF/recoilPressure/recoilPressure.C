#include "recoilPressure.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

recoilPressureModel::recoilPressureModel
(
    const fvMesh& mesh,
    const IOdictionary& thermo
)
:
    mesh_(mesh),
    thermoDict_(thermo)
{
    const dictionary& rp = thermo.subDict("recoilPressure");
    prefactor_ = rp.get<scalar>("prefactor");
    pAtm_ = rp.get<scalar>("pAtm");
    clampMin_ = rp.get<scalar>("clampMin");
    clampMax_ = rp.get<scalar>("clampMax");

    // Boiling point and latent heat of vaporization from thermophysicalProperties
    IOdictionary thermoProps
    (
        IOobject
        (
            "thermophysicalProperties",
            mesh_.time().constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    const dictionary& td = thermoProps.subDict("mixture").subDict("thermodynamics");
    Tb_ = td.get<scalar>("Tb");
    Lv_ = td.get<scalar>("Hv");

    // Effective molar mass of alloy vapor [kg/mol]
    M_ = 0.048;
    Rgas_ = 8.314;
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void recoilPressureModel::compute
(
    const volScalarField& T,
    volScalarField& P_r
) const
{
    forAll(T, cellI)
    {
        const scalar Ti = T[cellI];
        if (Ti > Tb_)
        {
            scalar Pr = prefactor_*pAtm_
                *Foam::exp(Lv_*M_*(Ti - Tb_)/(Rgas_*Ti*Tb_));
            P_r[cellI] = Foam::min(Foam::max(Pr, clampMin_), clampMax_);
        }
        else
        {
            P_r[cellI] = 0.0;
        }
    }
}


Foam::volVectorField recoilPressureModel::momentumSource
(
    const volScalarField& P_r,
    const volScalarField& alphaMetal
) const
{
    volVectorField source
    (
        IOobject
        (
            "recoilSource",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedVector(dimForce/dimVolume, Zero)
    );

    // CSF-like body force: P_r * grad(alpha) acts normal to the interface.
    // grad(alphaMetal) points from vapor into metal; recoil from expanding
    // vapor pushes liquid metal in this direction (down into the keyhole).
    source = P_r*fvc::grad(alphaMetal);

    // Limit to metal-containing cells (avoid forcing in pure shield gas)
    forAll(source, cellI)
    {
        if (alphaMetal[cellI] < 0.05)
        {
            source[cellI] = vector::zero;
        }
    }

    return source;
}

// ************************************************************************* //
