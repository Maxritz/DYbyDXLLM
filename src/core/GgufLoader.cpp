// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#include "dybydx/core/GgufLoader.h"
#include <iostream>
#include <cstring>
#include <functional>

namespace DirectLLM {

    GgufLoader::GgufLoader() : m_version(0), m_tensorCount(0), m_metadataCount(0), m_alignment(32), m_tensorDataStart(nullptr) {}

    GgufLoader::~GgufLoader() {
        Unload();
    }

    void GgufLoader::Unload() {
        m_metadata.clear();
        m_tensors.clear();
        m_fileBuffer.clear();
        m_tensorDataStart = nullptr;
    }

    bool GgufLoader::LoadFile(const std::string& filepath) {
        Unload();
        m_filepath = filepath;

        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[GgufLoader] FAILED to open file: " << filepath << std::endl;
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        m_fileBuffer.resize(fileSize);
        if (!file.read(reinterpret_cast<char*>(m_fileBuffer.data()), fileSize)) {
            std::cerr << "[GgufLoader] FAILED to read file to buffer: " << filepath << std::endl;
            return false;
        }
        file.close();

        size_t cursor = 0;
        auto readBytes = [&](void* dest, size_t size) -> bool {
            if (cursor + size > m_fileBuffer.size()) return false;
            std::memcpy(dest, &m_fileBuffer[cursor], size);
            cursor += size;
            return true;
        };

        char magic[4];
        if (!readBytes(magic, 4)) return false;
        if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
            std::cerr << "[GgufLoader] Invalid Magic: " << magic[0] << magic[1] << magic[2] << magic[3] << std::endl;
            return false;
        }

        if (!readBytes(&m_version, sizeof(m_version))) return false;

        if (m_version == 1) {
            uint32_t tCount, mCount;
            if (!readBytes(&tCount, sizeof(tCount))) return false;
            if (!readBytes(&mCount, sizeof(mCount))) return false;
            m_tensorCount = tCount;
            m_metadataCount = mCount;
        } else {
            if (!readBytes(&m_tensorCount, sizeof(m_tensorCount))) return false;
            if (!readBytes(&m_metadataCount, sizeof(m_metadataCount))) return false;
        }

        auto readStringLocal = [&]() -> std::string {
            uint64_t len = 0;
            if (m_version == 1) {
                uint32_t len32 = 0;
                if (!readBytes(&len32, sizeof(len32))) return "";
                len = len32;
            } else {
                if (!readBytes(&len, sizeof(len))) return "";
            }
            if (len == 0) return "";
            std::string s(len, '\0');
            readBytes(&s[0], len);
            return s;
        };

        std::function<GgufValue(GgufType)> readGgufValueLocal = [&](GgufType type) -> GgufValue {
            GgufValue val;
            val.Type = type;

            switch (type) {
                case GgufType::UINT8:
                case GgufType::INT8:
                case GgufType::BOOL:
                    val.Data.resize(1);
                    readBytes(val.Data.data(), 1);
                    break;
                case GgufType::UINT16:
                case GgufType::INT16:
                    val.Data.resize(2);
                    readBytes(val.Data.data(), 2);
                    break;
                case GgufType::UINT32:
                case GgufType::INT32:
                case GgufType::FLOAT32:
                    val.Data.resize(4);
                    readBytes(val.Data.data(), 4);
                    break;
                case GgufType::UINT64:
                case GgufType::INT64:
                case GgufType::FLOAT64:
                    val.Data.resize(8);
                    readBytes(val.Data.data(), 8);
                    break;
                case GgufType::STRING:
                    val.StringValue = readStringLocal();
                    break;
                case GgufType::ARRAY: {
                    uint32_t arrTypeVal = 0;
                    readBytes(&arrTypeVal, sizeof(arrTypeVal));
                    val.ArrayType = static_cast<GgufType>(arrTypeVal);

                    uint64_t arrLen = 0;
                    if (m_version == 1) {
                        uint32_t arrLen32 = 0;
                        readBytes(&arrLen32, sizeof(arrLen32));
                        arrLen = arrLen32;
                    } else {
                        readBytes(&arrLen, sizeof(arrLen));
                    }
                    val.ArrayLength = arrLen;
                    val.ArrayValues.reserve(arrLen);
                    for (uint64_t i = 0; i < arrLen; ++i) {
                        val.ArrayValues.push_back(readGgufValueLocal(val.ArrayType));
                    }
                    break;
                }
            }
            return val;
        };

        for (uint64_t i = 0; i < m_metadataCount; ++i) {
            std::string key = readStringLocal();
            uint32_t typeVal = 0;
            if (!readBytes(&typeVal, sizeof(typeVal))) return false;
            
            GgufType valType = static_cast<GgufType>(typeVal);
            GgufValue val = readGgufValueLocal(valType);
            m_metadata[key] = val;

            if (key == "general.alignment") {
                if (val.Type == GgufType::UINT32) {
                    m_alignment = *reinterpret_cast<const uint32_t*>(val.Data.data());
                } else if (val.Type == GgufType::UINT64) {
                    m_alignment = *reinterpret_cast<const uint64_t*>(val.Data.data());
                }
            }
        }

        for (uint64_t i = 0; i < m_tensorCount; ++i) {
            GgufTensor tensor;
            tensor.Name = readStringLocal();

            uint32_t dimsCount = 0;
            if (!readBytes(&dimsCount, sizeof(dimsCount))) return false;

            tensor.Shape.resize(dimsCount);
            for (uint32_t d = 0; d < dimsCount; ++d) {
                uint64_t dimValue = 0;
                if (m_version == 1) {
                    uint32_t dimValue32 = 0;
                    if (!readBytes(&dimValue32, sizeof(dimValue32))) return false;
                    dimValue = dimValue32;
                } else {
                    if (!readBytes(&dimValue, sizeof(dimValue))) return false;
                }
                tensor.Shape[d] = dimValue;
            }

            uint32_t typeVal = 0;
            if (!readBytes(&typeVal, sizeof(typeVal))) return false;
            tensor.Type = static_cast<GgmlType>(typeVal);

            if (!readBytes(&tensor.FileOffset, sizeof(tensor.FileOffset))) return false;

            uint64_t elementsCount = 1;
            for (auto dim : tensor.Shape) elementsCount *= dim;

            switch (tensor.Type) {
                case GgmlType::F32: tensor.DataSize = elementsCount * 4; break;
                case GgmlType::F16: tensor.DataSize = elementsCount * 2; break;
                case GgmlType::I8:  tensor.DataSize = elementsCount; break;
                case GgmlType::I16: tensor.DataSize = elementsCount * 2; break;
                case GgmlType::I32: tensor.DataSize = elementsCount * 4; break;
                case GgmlType::Q4_0: {
                    tensor.DataSize = (elementsCount / 32) * 18;
                    break;
                }
                case GgmlType::Q8_0: {
                    tensor.DataSize = (elementsCount / 32) * 34;
                    break;
                }
                default:
                    tensor.DataSize = elementsCount * 2;
                    break;
            }

            m_tensors[tensor.Name] = tensor;
        }

        size_t alignmentPadding = (cursor + m_alignment - 1) & ~(m_alignment - 1);
        if (alignmentPadding <= m_fileBuffer.size()) {
            m_tensorDataStart = m_fileBuffer.data() + alignmentPadding;
        } else {
            return false;
        }

        for (auto& pair : m_tensors) {
            pair.second.DataPtr = m_tensorDataStart + pair.second.FileOffset;
        }

        return true;
    }

    bool GgufLoader::HasMetadata(const std::string& key) const {
        return m_metadata.find(key) != m_metadata.end();
    }

    std::string GgufLoader::GetMetadataString(const std::string& key, const std::string& defaultVal) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() && it->second.Type == GgufType::STRING) {
            return it->second.StringValue;
        }
        return defaultVal;
    }

    uint32_t GgufLoader::GetMetadataUint32(const std::string& key, uint32_t defaultVal) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end()) {
            if (it->second.Type == GgufType::UINT32 || it->second.Type == GgufType::INT32) {
                return *reinterpret_cast<const uint32_t*>(it->second.Data.data());
            }
        }
        return defaultVal;
    }

    uint64_t GgufLoader::GetMetadataUint64(const std::string& key, uint64_t defaultVal) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end()) {
            if (it->second.Type == GgufType::UINT64 || it->second.Type == GgufType::INT64) {
                return *reinterpret_cast<const uint64_t*>(it->second.Data.data());
            }
        }
        return defaultVal;
    }

    float GgufLoader::GetMetadataFloat(const std::string& key, float defaultVal) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() && it->second.Type == GgufType::FLOAT32) {
            return *reinterpret_cast<const float*>(it->second.Data.data());
        }
        return defaultVal;
    }

    const GgufTensor* GgufLoader::GetTensor(const std::string& name) const {
        auto it = m_tensors.find(name);
        if (it != m_tensors.end()) {
            return &it->second;
        }
        return nullptr;
    }
}
