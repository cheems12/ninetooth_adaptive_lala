local PYBIND11_INCLUDE = os.getenv("INFINILM_PYBIND11_INCLUDE")
local PYTHON_INCLUDE = os.getenv("INFINILM_PYTHON_INCLUDE")
if not PYBIND11_INCLUDE or PYBIND11_INCLUDE == "" then
    add_requires("pybind11")
end

set_toolchains("gcc")

option("cxx11-abi")
    set_default(nil)
    set_showmenu(true)
    set_description("Set _GLIBCXX_USE_CXX11_ABI to match the installed InfiniCore")
    set_values("0", "1")
option_end()

local cxx11_abi = os.getenv("INFINILM_CXX11_ABI") or get_config("cxx11-abi")
if cxx11_abi and cxx11_abi ~= "" then
    if cxx11_abi ~= "0" and cxx11_abi ~= "1" then
        raise("INFINILM_CXX11_ABI must be 0 or 1")
    end
    add_defines("_GLIBCXX_USE_CXX11_ABI=" .. cxx11_abi)
end

-- Add spdlog from third_party directory
add_includedirs("third_party/spdlog/include")
add_includedirs("third_party/json/single_include/")

target("_infinilm")
    if PYBIND11_INCLUDE and PYBIND11_INCLUDE ~= "" then
        add_includedirs(PYBIND11_INCLUDE)
        if PYTHON_INCLUDE and PYTHON_INCLUDE ~= "" then
            add_includedirs(PYTHON_INCLUDE)
        end
        -- The python.module rule is provided by xmake's pybind11 package.
        -- Offline accelerator images may only provide the headers (PyTorch
        -- vendors them), so emit a directly importable extension instead.
        set_filename("_infinilm.so")
        set_targetdir("python/infinilm/lib")
    else
        add_packages("pybind11")
        add_rules("python.module", {soabi = true})
    end
    set_default(false)
    set_languages("cxx17")
    set_kind("shared")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    -- add_includedirs("csrc", { public = false })
    -- add_includedirs("csrc/pybind11", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl")

    -- Add C++ sources
    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")

    if PYBIND11_INCLUDE and PYBIND11_INCLUDE ~= "" then
        -- Keep install output separate from set_targetdir above.  Copying a
        -- target onto itself truncates the extension on xmake 2.9.x.
        set_installdir("build/infinilm-install")
    else
        set_installdir("python/infinilm")
    end
target_end()
