cmake_minimum_required(VERSION 3.20)
project(DybyDx VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CORE_SOURCES
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

add_executable(DybyDx
    cli/main.cpp
    ${CORE_SOURCES}
)

target_include_directories(DybyDx PRIVATE
    include
    ${DIRECTSTORAGE_DIR}/native/include
)

target_link_directories(DybyDx PRIVATE
    ${DIRECTSTORAGE_DIR}/native/lib/x64
)

target_link_libraries(DybyDx PRIVATE
    d3d12 dxgi d3dcompiler dstorage
)

add_custom_command(TARGET DybyDx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "$<TARGET_FILE_DIR:DybyDx>/shaders"
    COMMENT "Copying HLSL shaders to output..."
)

add_custom_command(TARGET DybyDx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DIRECTSTORAGE_DIR}/native/bin/x64/dstorage.dll"
        "$<TARGET_FILE_DIR:DybyDx>/dstorage.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DIRECTSTORAGE_DIR}/native/bin/x64/dstoragecore.dll"
        "$<TARGET_FILE_DIR:DybyDx>/dstoragecore.dll"
    COMMENT "Copying DirectStorage runtime DLLs..."
)

message(STATUS "DybyDx configured. Target: DybyDx.exe")
