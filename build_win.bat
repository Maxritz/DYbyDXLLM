@echo off
set VULKAN_SDK=C:\VulkanSDK\1.4.350.0
mkdir build 2>nul
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
echo Done! Check bin/Release/ for DybyDx.dll
rem pause
