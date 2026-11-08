"""Generate the InfiniCore-side NineToothed (AOT) dispatcher for one operator.

This module drives NineToothed's compiler directly so that every specialization
produces three things inside ``build/ninetoothed``:

* ``<kernel>.triton.so``          -- the reloadable Triton host launcher;
* ``<kernel>.triton.*.cpp``       -- the C++ sources of that launcher, including
  NineToothed's ``wrapper_source`` adaptor which exports
  ``NineToothedResult launch_<kernel>_variant(NineToothedStream, NineToothedTensor...)``;
* ``<kernel>.h``                  -- the declaration of the exported symbol.

``xmake/hygon.lua`` compiles ``build/ninetoothed/*.c`` and ``*.cpp`` into
``infiniop-hygon``, and the operator wrapper (e.g.
``src/infiniop/ops/mrope/ninetoothed/mrope.h``) calls the dispatcher generated
here as ``launch_<op>(stream, <tensor>..., <config int>...)``.

The tensor order of that dispatcher is exactly ``compilation.launch_abi.public_args``,
which is also the order NineToothed's adaptor expects.
"""

import concurrent.futures
import functools
import itertools
import pathlib

from ninetoothed.backends.materializers.triton import publish_cpp_sources
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest

CURRENT_FILE_PATH = pathlib.Path(__file__)

BUILD_DIRECTORY_PATH = (
    CURRENT_FILE_PATH.parent.parent.parent.parent / "build" / "ninetoothed"
)


def build(premake, constexpr_param_grid, caller, op_name, output_dir):
    combinations_to_build = tuple(
        _generate_param_value_combinations(constexpr_param_grid)
    )
    max_workers = min(1, max(1, len(combinations_to_build)))

    if not combinations_to_build:
        raise ValueError(f"{op_name}: the constexpr parameter grid is empty")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_make, premake, combination, caller, op_name, output_dir)
            for combination in combinations_to_build
        ]

        # Preserve grid order so generated dispatch code is reproducible.
        results = [future.result() for future in futures]

    headers = [result[0] for result in results]
    all_param_names = [result[1] for result in results]
    combinations = [result[2] for result in results]
    launches = [result[3] for result in results]

    # Every specialization of one operator shares the same public tensor order,
    # so the dispatcher signature comes from the first specialization.
    param_names = list(all_param_names[0])
    param_types = ["NineToothedStream"] + [
        "NineToothedTensor" for _ in range(len(param_names) - 1)
    ]

    for param_name in functools.reduce(lambda x, y: x | y, combinations, {}):
        param_names.append(param_name)
        param_types.append("int")

    param_decls = ", ".join(
        f"{type_} {param}" for param, type_ in zip(param_names, param_types)
    )

    source_file_name = f"{op_name}.c"
    header_file_name = f"{op_name}.h"

    func_sig = f"NineToothedResult launch_{op_name}({param_decls})"

    joined_launches = "\n".join(launches)

    op_decl = f'#ifdef __cplusplus\nextern "C" {func_sig};\n#else\n{func_sig};\n#endif'
    op_def = f"""{func_sig} {{
{joined_launches}
    return INFINI_STATUS_NOT_IMPLEMENTED;
}}"""

    includes = "\n".join(f'#include "{header}"' for header in headers)
    source_content = f"""#include "{header_file_name}"

#include "infinicore.h"

{includes}\n\n{op_def}\n"""
    header_content = f'#include "ninetoothed.h"\n\n{op_decl}\n'

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / source_file_name).write_text(source_content)
    (output_dir / header_file_name).write_text(header_content)


def _make(premake, combination, caller, op_name, output_dir):
    arrangement, application, tensors = premake(**combination)

    for param_name, param_value in combination.items():
        if isinstance(param_value, str):
            combination[param_name] = (
                f"INFINI_DTYPE_{combination[param_name].replace('fp', 'F').upper()}"
            )

    kernel_name = f"{op_name}_{_generate_suffix(combination.values())}"
    combination = {f"{name}_": value for name, value in combination.items()}

    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tuple(tensors),
            caller=caller,
            kernel_name=kernel_name,
            max_num_configs=1,
        )
    )

    # Publishes <kernel_name>.triton.so into `output_dir` and materializes the
    # generated Triton AOT library in the NineToothed cache.
    DEFAULT_COMPILER.materialize(compilation, output_dir=output_dir, mode="aot")

    # Publishes <kernel_name>.triton.{compiled,launch,linked}.cpp plus the shared
    # ninetoothed.h.  The `launch.cpp` file carries NineToothed's ABI adaptor,
    # which is what defines `launch_<kernel_name>_variant`.
    publish_cpp_sources(compilation, output_dir)

    public_args = tuple(compilation.launch_abi.public_args)

    header = _write_kernel_header(kernel_name, public_args, output_dir)

    # The dispatcher's own parameters are named after `public_args`, in the same
    # order, so forward them positionally to NineToothed's ABI adaptor.
    launch = f"""    if ({_generate_condition(combination)})
        return launch_{kernel_name}_variant({", ".join(("stream", *public_args))});"""

    return header, ("stream", *public_args), combination, launch


def _write_kernel_header(kernel_name, public_args, output_dir):
    """Declare the symbol exported by NineToothed's Triton AOT adaptor."""
    parameters = ["NineToothedStream stream"] + [
        f"NineToothedTensor tensor_{index}" for index in range(len(public_args))
    ]
    declaration = (
        f"NineToothedResult launch_{kernel_name}_variant({', '.join(parameters)})"
    )

    path = pathlib.Path(output_dir) / f"{kernel_name}.h"
    path.write_text(
        "#include \"ninetoothed.h\"\n\n"
        "#ifdef __cplusplus\n"
        f'extern "C" {declaration};\n'
        "#else\n"
        f"{declaration};\n"
        "#endif\n"
    )

    return path


def _generate_condition(combination):
    return " && ".join(
        f"{param} == {_condition_literal(value)}"
        for param, value in combination.items()
    )


def _condition_literal(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _generate_suffix(values):
    return "_".join(f"{value}" for value in values)


def _generate_param_value_combinations(param_grid):
    keys = list(param_grid.keys())
    value_combinations = itertools.product(*param_grid.values())

    return tuple(dict(zip(keys, combination)) for combination in value_combinations)
