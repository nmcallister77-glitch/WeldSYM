    Info<< "Reading field p_rgh\n" << endl;
    volScalarField p_rgh
    (
        IOobject
        (
            "p_rgh",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    Info<< "Reading field U\n" << endl;
    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    #include "createPhi.H"

    Info<< "Reading transportProperties\n" << endl;
    immiscibleIncompressibleTwoPhaseMixture mixture(U, phi);

    volScalarField& alpha1(mixture.alpha1());
    volScalarField& alpha2(mixture.alpha2());

    alpha1.rename("alpha.metal");
    alpha2.rename("alpha.vapor");

    // Reference phase densities from transportProperties
    const dimensionedScalar& rho1 = mixture.rho1();
    const dimensionedScalar& rho2 = mixture.rho2();

    // Mixture density and mass flux (updated after alpha transport)
    volScalarField rho
    (
        IOobject
        (
            "rho",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        alpha1*rho1 + alpha2*rho2
    );

    surfaceScalarField rhoPhi
    (
        IOobject
        (
            "rhoPhi",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        fvc::interpolate(rho)*phi
    );

    surfaceScalarField rhoCpPhi
    (
        IOobject
        (
            "rhoCpPhi",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        fvc::interpolate(rho*Cp)*phi
    );

    // Thermophysical properties (constant with T for now; use tabulated later)
    IOdictionary thermoDict
    (
        IOobject
        (
            "thermophysicalProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const dimensionedScalar Cp
    (
        "Cp",
        dimEnergy/dimMass/dimTemperature,
        thermoDict.subDict("mixture").subDict("thermodynamics").get<scalar>("Cp")
    );

    const dimensionedScalar kappa
    (
        "kappa",
        dimPower/dimLength/dimTemperature,
        thermoDict.subDict("mixture").subDict("transport").get<scalar>("kappa")
    );

    volScalarField alpha
    (
        IOobject
        (
            "alpha",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        kappa/(rho*Cp)
    );

    volScalarField kEff
    (
        IOobject
        (
            "kEff",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        rho*Cp*alpha
    );

    Info<< "Reading field T\n" << endl;
    volScalarField T
    (
        IOobject
        (
            "T",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    volScalarField liquidFraction
    (
        IOobject
        (
            "liquidFraction",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar(dimless, Zero)
    );

    volScalarField recoilPressure
    (
        IOobject
        (
            "recoilPressure",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar(dimPressure, Zero)
    );

    volScalarField laserHeatFlux
    (
        IOobject
        (
            "laserHeatFlux",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar(dimPower/dimVolume, Zero)
    );

    #include "readGravitationalAcceleration.H"
    #include "readhRef.H"
    #include "gh.H"

    volScalarField p
    (
        IOobject
        (
            "p",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        p_rgh + rho*gh
    );

    label pRefCell = 0;
    scalar pRefValue = 0.0;
    setRefCell
    (
        p,
        p_rgh,
        pimple.dict(),
        pRefCell,
        pRefValue
    );

    if (p_rgh.needReference())
    {
        p += dimensionedScalar
        (
            "p",
            p.dimensions(),
            pRefValue - getRefCellValue(p, pRefCell)
        );
        p_rgh = p - rho*gh;
    }

    mesh.setFluxRequired(p_rgh.name());
    mesh.setFluxRequired(alpha1.name());

    #include "createMRF.H"
    #include "createFvOptions.H"
    #include "createTurbulence.H"

    IOdictionary laserDict
    (
        IOobject
        (
            "laserHeatSource",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
