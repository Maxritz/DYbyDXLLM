cmake_minimum_required(VERSION 3.20)
project(DybyDx VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

set(DIRECTSTORAGE_DIR "C:/Users/rina0423/Desktop/DX/directstorage")
set(DXC_DIR "C:/Users/rina0423/Desktop/DX/DXcompiler")
set(OPENVINO_DIR "C:/Users/rina0423/Desktop/DX/openvino/runtime")

# OpenVINO SDK paths
set(OPENVINO_INCLUDE_DIR "${OPENVINO_DIR}/include")
set(OPENVINO_LIB_DIR "${OPENVINO_DIR}/lib/intel64/Release")

# Native DLL for C# CLI P/Invoke
add_library(DirectLLM.Core SHARED ${SOURCES})
target_include_directories(DirectLLM.Core PRIVATE
    include
    ${DIRECTSTORAGE_DIR}/native/include
    ${DXC_DIR}/inc
)
target_link_directories(DirectLLM.Core PRIVATE
    ${DIRECTSTORAGE_DIR}/native/lib/x64
    ${DXC_DIR}/lib/x64
)
target_link_libraries(DirectLLM.Core PRIVATE d3d12 dxgi d3dcompiler dstorage dxcompiler)

add_executable(DybyDx cli/main.cpp ${SOURCES})
target_include_directories(DybyDx PRIVATE
    include
    ${DIRECTSTORAGE_DIR}/native/include
    ${DXC_DIR}/inc
)
target_link_directories(DybyDx PRIVATE
    ${DIRECTSTORAGE_DIR}/native/lib/x64
    ${DXC_DIR}/lib/x64
)
target_link_libraries(DybyDx PRIVATE d3d12 dxgi d3dcompiler dstorage dxcompiler)

# Post-build: copy shaders and DirectStorage DLLs
foreach(TGT DirectLLM.Core DybyDx)
    add_custom_command(TARGET ${TGT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "$<TARGET_FILE_DIR:${TGT}>/shaders"
        COMMENT "Copying HLSL shaders to ${TGT} output..."
    )
    add_custom_command(TARGET ${TGT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DIRECTSTORAGE_DIR}/native/bin/x64/dstorage.dll"
            "$<TARGET_FILE_DIR:${TGT}>/dstorage.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DIRECTSTORAGE_DIR}/native/bin/x64/dstoragecore.dll"
            "$<TARGET_FILE_DIR:${TGT}>/dstoragecore.dll"
        COMMENT "Copying DirectStorage runtime DLLs to ${TGT}..."
    )
    add_custom_command(TARGET ${TGT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DXC_DIR}/bin/x64/dxcompiler.dll"
            "$<TARGET_FILE_DIR:${TGT}>/dxcompiler.dll"
        COMMENT "Copying DXC compiler DLL to ${TGT}..."
    )
endforeach()

message(STATUS "DybyDx configured. Targets: DirectLLM.Core.dll + DybyDx.exe")
