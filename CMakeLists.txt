cmake_minimum_required(VERSION 3.20)
project(DybyDx VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(NOMINMAX)
add_compile_definitions(ZEN3_OPT=1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(SOURCES
    src/core/DirectXEngine.cpp
    src/core/KVCacheManager.cpp
    src/core/BpeTokenizer.cpp
    src/core/DirectStorageLoader.cpp
    src/core/IntelOpenVINOInterop.cpp
    src/core/AdvancedVendorOptimizations.cpp
    src/core/ModelPipeline.cpp
    src/core/GgufLoader.cpp
)

set(DIRECTSTORAGE_DIR "C:/Users/rr/Desktop/dxxxx/ds14")
set(DXC_DIR "C:/Users/rr/Desktop/dxxxx/dxc-1.10.2605.2")
set(OPENVINO_DIR "C:/Users/rr/Desktop/dxxxx/OpenVINO/runtime")

# Check if SDKs exist
if(EXISTS "${DIRECTSTORAGE_DIR}/native/include/dstorage.h")
    set(HAVE_DIRECTSTORAGE TRUE)
else()
    set(HAVE_DIRECTSTORAGE FALSE)
    message(WARNING "DirectStorage not found - building without DirectStorage")
endif()
if(EXISTS "${DXC_DIR}/inc/dxcapi.h")
    set(HAVE_DXC TRUE)
else()
    set(HAVE_DXC FALSE)
    message(WARNING "DXC not found - building without DXC shader compilation")
endif()
if(EXISTS "${OPENVINO_DIR}/include/openvino.hpp")
    set(HAVE_OPENVINO TRUE)
else()
    set(HAVE_OPENVINO FALSE)
    message(WARNING "OpenVINO not found - building without OpenVINO")
endif()

# Compiler optimisation + SIMD flags
if(MSVC)
    add_compile_options(
        /O2           # Full optimisation
        /Oi           # Intrinsic functions
        /Ot           # Favour speed over size
        /fp:fast      # Fast floating-point
        /arch:AVX2    # Enable AVX2 intrinsic headers
    )
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

set(OPENVINO_INCLUDE_DIR "${OPENVINO_DIR}/include")
set(OPENVINO_LIB_DIR "${OPENVINO_DIR}/lib/intel64/Release")

# Native DLL for C# CLI P/Invoke
add_library(DirectLLM.Core SHARED ${SOURCES})
target_include_directories(DirectLLM.Core PRIVATE
    include
)
if(HAVE_OPENVINO)
    target_include_directories(DirectLLM.Core PRIVATE ${OPENVINO_INCLUDE_DIR})
endif()
if(HAVE_DIRECTSTORAGE)
    target_include_directories(DirectLLM.Core PRIVATE ${DIRECTSTORAGE_DIR}/native/include)
endif()
if(HAVE_DXC)
    target_include_directories(DirectLLM.Core PRIVATE ${DXC_DIR}/inc)
endif()
if(HAVE_OPENVINO)
    target_link_directories(DirectLLM.Core PRIVATE ${OPENVINO_LIB_DIR})
endif()
if(HAVE_DIRECTSTORAGE)
    target_link_directories(DirectLLM.Core PRIVATE ${DIRECTSTORAGE_DIR}/native/lib/x64)
endif()
if(HAVE_DXC)
    target_link_directories(DirectLLM.Core PRIVATE ${DXC_DIR}/lib/x64)
endif()

set(LINK_LIBS d3d12 dxgi d3dcompiler dxcompiler)
if(HAVE_OPENVINO)
    list(APPEND LINK_LIBS openvino)
endif()
target_link_libraries(DirectLLM.Core PRIVATE ${LINK_LIBS})
if(HAVE_DIRECTSTORAGE)
    target_link_libraries(DirectLLM.Core PRIVATE dstorage)
endif()

add_executable(DybyDx cli/main.cpp ${SOURCES})
target_include_directories(DybyDx PRIVATE include)
if(HAVE_OPENVINO)
    target_include_directories(DybyDx PRIVATE ${OPENVINO_INCLUDE_DIR})
endif()
if(HAVE_DIRECTSTORAGE)
    target_include_directories(DybyDx PRIVATE ${DIRECTSTORAGE_DIR}/native/include)
endif()
if(HAVE_DXC)
    target_include_directories(DybyDx PRIVATE ${DXC_DIR}/inc)
endif()
if(HAVE_OPENVINO)
    target_link_directories(DybyDx PRIVATE ${OPENVINO_LIB_DIR})
endif()
if(HAVE_DIRECTSTORAGE)
    target_link_directories(DybyDx PRIVATE ${DIRECTSTORAGE_DIR}/native/lib/x64)
endif()
if(HAVE_DXC)
    target_link_directories(DybyDx PRIVATE ${DXC_DIR}/lib/x64)
endif()
target_link_libraries(DybyDx PRIVATE ${LINK_LIBS})
if(HAVE_DIRECTSTORAGE)
    target_link_libraries(DybyDx PRIVATE dstorage)
endif()

# Post-build: copy shaders
foreach(TGT DirectLLM.Core DybyDx)
    add_custom_command(TARGET ${TGT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "$<TARGET_FILE_DIR:${TGT}>/shaders"
        COMMENT "Copying HLSL shaders to ${TGT} output..."
    )
    if(HAVE_DIRECTSTORAGE)
        add_custom_command(TARGET ${TGT} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DIRECTSTORAGE_DIR}/native/bin/x64/dstorage.dll"
                "$<TARGET_FILE_DIR:${TGT}>/dstorage.dll"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DIRECTSTORAGE_DIR}/native/bin/x64/dstoragecore.dll"
                "$<TARGET_FILE_DIR:${TGT}>/dstoragecore.dll"
            COMMENT "Copying DirectStorage runtime DLLs to ${TGT}..."
        )
    endif()
    if(HAVE_DXC)
        add_custom_command(TARGET ${TGT} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DXC_DIR}/bin/x64/dxcompiler.dll"
                "$<TARGET_FILE_DIR:${TGT}>/dxcompiler.dll"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DXC_DIR}/bin/x64/dxil.dll"
                "$<TARGET_FILE_DIR:${TGT}>/dxil.dll"
            COMMENT "Copying DXC compiler DLLs to ${TGT}..."
        )
    endif()
    # Copy DirectX 12 SDK layers for graphics debugging
    add_custom_command(TARGET ${TGT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/../sdk/build/native/bin/x64/d3d12SDKLayers.dll"
            "$<TARGET_FILE_DIR:${TGT}>/d3d12SDKLayers.dll"
        COMMENT "Copying D3D12 SDK Layers to ${TGT}..."
    )
endforeach()

message(STATUS "DybyDx configured. Targets: DirectLLM.Core.dll + DybyDx.exe")