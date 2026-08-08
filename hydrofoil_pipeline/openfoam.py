from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from hydrofoil_pipeline.cases import Case
from hydrofoil_pipeline.naca import coordinates

DOCKER_IMAGE = "openfoam/openfoam9-graphical-apps"
CONTAINER_ROOT = Path("/home/openfoam")


def write_case(case: Case, case_dir: Path, airfoil_coords=None) -> None:
    """Write a minimal simpleFoam-ready case skeleton."""
    for sub in ["0", "constant", "system"]:
        (case_dir / sub).mkdir(parents=True, exist_ok=True)

    ax, ay = airfoil_coords if airfoil_coords is not None else coordinates(case.naca, n=121, chord=case.chord)
    points = "\n".join(f"{x:.8f} {y:.8f} 0" for x, y in zip(ax, ay))
    (case_dir / "airfoil_points.dat").write_text(points + "\n")

    (case_dir / "case_parameters.txt").write_text(
        "\n".join(
            [
                f"naca {case.naca}",
                f"AoA_deg {case.AoA}",
                f"Re {case.Re}",
                f"U_inf {case.U_inf}",
                f"rho {case.rho}",
                f"nu {case.nu}",
                f"p_inf {case.p_inf}",
                f"p_vap {case.p_vap}",
            ]
        )
        + "\n"
    )

    _write_gmsh_geo(case, case_dir, ax, ay)
    (case_dir / "constant" / "transportProperties").write_text(
        f"""{_foam_header("dictionary", "transportProperties")}
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] {case.nu};
"""
    )
    (case_dir / "constant" / "turbulenceProperties").write_text(
        f"""{_foam_header("dictionary", "turbulenceProperties")}
simulationType RAS;
RAS
{{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}}
"""
    )
    _write_system_files(case, case_dir)
    _write_initial_fields(case, case_dir)


def openfoam_available(use_docker: bool = False) -> bool:
    if use_docker:
        return shutil.which("docker") is not None
    return shutil.which("simpleFoam") is not None


def run_openfoam_case(case_dir: Path, repo_root: Path, use_docker: bool = False) -> None:
    generate_mesh(case_dir, repo_root, use_docker=use_docker)
    command = (
        f"checkMesh -case {container_path(case_dir, repo_root)} > {container_path(case_dir, repo_root)}/log.checkMesh"
        f" && simpleFoam -case {container_path(case_dir, repo_root)} > {container_path(case_dir, repo_root)}/log.simpleFoam"
        f" && postProcess -case {container_path(case_dir, repo_root)} -latestTime -func writeCellCentres > {container_path(case_dir, repo_root)}/log.writeCellCentres"
    )
    run_openfoam_command(command, repo_root, use_docker=use_docker)


def generate_mesh(case_dir: Path, repo_root: Path, use_docker: bool = False) -> None:
    geo = case_dir / "mesh.geo"
    msh = case_dir / "mesh.msh"
    subprocess.run(["gmsh", "-3", "-format", "msh2", str(geo), "-o", str(msh)], check=True)
    command = f"gmshToFoam -case {container_path(case_dir, repo_root)} {container_path(msh, repo_root)} > {container_path(case_dir, repo_root)}/log.gmshToFoam"
    run_openfoam_command(command, repo_root, use_docker=use_docker)
    set_boundary_type(case_dir / "constant" / "polyMesh" / "boundary", "frontAndBack", "empty")


def set_boundary_type(boundary_path: Path, patch_name: str, patch_type: str) -> None:
    text = boundary_path.read_text()
    marker = f"    {patch_name}\n    {{"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Patch {patch_name!r} was not found in {boundary_path}")
    end = text.find("    }\n", start)
    if end < 0:
        raise RuntimeError(f"Patch {patch_name!r} block was not closed in {boundary_path}")
    block = text[start:end]
    block = block.replace("type            patch;", f"type            {patch_type};")
    block = block.replace("physicalType    patch;", f"physicalType    {patch_type};")
    boundary_path.write_text(text[:start] + block + text[end:])


def run_openfoam_command(command: str, repo_root: Path, use_docker: bool = False) -> None:
    if use_docker:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "bash",
                "-v",
                f"{repo_root.resolve()}:{CONTAINER_ROOT}",
                "-w",
                str(CONTAINER_ROOT),
                DOCKER_IMAGE,
                "-lc",
                f"source /opt/openfoam9/etc/bashrc && {command}",
            ],
            check=True,
        )
        return

    if not shutil.which("simpleFoam"):
        raise RuntimeError("simpleFoam was not found on PATH. Source OpenFOAM or use Docker mode.")
    subprocess.run(["bash", "-lc", command], check=True)


def container_path(path: Path, repo_root: Path) -> str:
    rel = path.resolve().relative_to(repo_root.resolve())
    return str(CONTAINER_ROOT / rel)


def _write_gmsh_geo(case: Case, case_dir: Path, ax, ay) -> None:
    far_x0 = -20.0 * case.chord
    far_x1 = 20.0 * case.chord
    far_y0 = -12.0 * case.chord
    far_y1 = 12.0 * case.chord
    lines = [
        'SetFactory("Built-in");',
        "Mesh.MshFileVersion = 2.2;",
        "lcFar = 1.0;",
        "lcFoil = 0.015;",
        f"Point(1) = {{{far_x0}, {far_y0}, 0, lcFar}};",
        f"Point(2) = {{{far_x1}, {far_y0}, 0, lcFar}};",
        f"Point(3) = {{{far_x1}, {far_y1}, 0, lcFar}};",
        f"Point(4) = {{{far_x0}, {far_y1}, 0, lcFar}};",
        "Line(1) = {1, 2};",
        "Line(2) = {2, 3};",
        "Line(3) = {3, 4};",
        "Line(4) = {4, 1};",
    ]
    start = 100
    for i, (x, y) in enumerate(zip(ax[:-1], ay[:-1]), start=start):
        lines.append(f"Point({i}) = {{{x:.8f}, {y:.8f}, 0, lcFoil}};")
    point_ids = list(range(start, start + len(ax) - 1))
    leading_edge = int(min(range(len(ax) - 1), key=lambda idx: ax[idx]))
    upper_ids = point_ids[: leading_edge + 1]
    lower_ids = point_ids[leading_edge:] + [point_ids[0]]
    lines.extend(
        [
            f"Spline(20) = {{{', '.join(map(str, upper_ids))}}};",
            f"Spline(21) = {{{', '.join(map(str, lower_ids))}}};",
            "Curve Loop(30) = {1, 2, 3, 4};",
            "Curve Loop(31) = {20, 21};",
            "Plane Surface(40) = {30, 31};",
            "out[] = Extrude {0, 0, 0.01} { Surface{40}; Layers{1}; Recombine; };",
            'Physical Surface("frontAndBack") = {40, out[0]};',
            'Physical Surface("topAndBottom") = {out[2], out[4]};',
            'Physical Surface("outlet") = {out[3]};',
            'Physical Surface("inlet") = {out[5]};',
            'Physical Surface("airfoil") = {out[6], out[7]};',
            'Physical Volume("internal") = {out[1]};',
            "Mesh.Algorithm = 6;",
        ]
    )
    (case_dir / "mesh.geo").write_text("\n".join(lines) + "\n")


def _write_system_files(case: Case, case_dir: Path) -> None:
    alpha = math.radians(case.AoA)
    drag_x = math.cos(alpha)
    drag_y = math.sin(alpha)
    lift_x = -math.sin(alpha)
    lift_y = math.cos(alpha)
    (case_dir / "system" / "controlDict").write_text(
        _foam_header("dictionary", "controlDict")
        + f"""application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         300;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        patches         (airfoil);
        writeControl    timeStep;
        writeInterval   10;
        p               p;
        U               U;
        rho             rhoInf;
        rhoInf          {case.rho:.12g};
        pRef            0;
        liftDir         ({lift_x:.12g} {lift_y:.12g} 0);
        dragDir         ({drag_x:.12g} {drag_y:.12g} 0);
        CofR            (0.25 0 0);
        pitchAxis       (0 0 1);
        magUInf         {case.U_inf:.12g};
        lRef            {case.chord:.12g};
        Aref            {case.chord * 0.01:.12g};
    }}
}}
"""
    )
    (case_dir / "system" / "fvSchemes").write_text(
        _foam_header("dictionary", "fvSchemes")
        + """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss upwind;
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
"""
    )
    (case_dir / "system" / "fvSolution").write_text(
        _foam_header("dictionary", "fvSolution")
        + """solvers
{
    p { solver GAMG; tolerance 1e-7; relTol 0.01; smoother GaussSeidel; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
    k { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
    omega { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent yes;
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; k 0.7; omega 0.7; }
}
"""
    )


def _write_initial_fields(case: Case, case_dir: Path) -> None:
    alpha = math.radians(case.AoA)
    ux = case.U_inf * math.cos(alpha)
    uy = case.U_inf * math.sin(alpha)
    (case_dir / "0" / "U").write_text(
        f"""{_foam_header("volVectorField", "U", location="0")}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({ux:.8g} {uy:.8g} 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform ({ux:.8g} {uy:.8g} 0); }}
    outlet {{ type zeroGradient; }}
    topAndBottom {{ type fixedValue; value uniform ({ux:.8g} {uy:.8g} 0); }}
    airfoil {{ type noSlip; }}
    frontAndBack {{ type empty; }}
}}
"""
    )
    (case_dir / "0" / "p").write_text(
        f"""{_foam_header("volScalarField", "p", location="0")}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform 0; }}
    topAndBottom {{ type zeroGradient; }}
    airfoil {{ type zeroGradient; }}
    frontAndBack {{ type empty; }}
}}
"""
    )
    dimensions = {
        "k": "[0 2 -2 0 0 0 0]",
        "omega": "[0 0 -1 0 0 0 0]",
        "nut": "[0 2 -1 0 0 0 0]",
    }
    for name, value in {"k": 1e-6, "omega": 1.0, "nut": 1e-8}.items():
        (case_dir / "0" / name).write_text(
            f"""{_foam_header("volScalarField", name, location="0")}
dimensions      {dimensions[name]};
internalField   uniform {value};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {value}; }}
    outlet {{ type zeroGradient; }}
    topAndBottom {{ type zeroGradient; }}
    airfoil {{ type fixedValue; value uniform {value}; }}
    frontAndBack {{ type empty; }}
}}
"""
        )


def _foam_header(class_name: str, object_name: str, location: str | None = None) -> str:
    location_line = f'    location    "{location}";\n' if location is not None else ""
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
{location_line}    object      {object_name};
}}

"""
