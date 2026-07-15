// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#include "dybydx/core/GgufLoader.h"
#include <iostream>
#include <cstring>
#include <functional>

namespace DirectLLM {

    GgufLoader::GgufLoader()
        : m_version(0), m_tensorCount(0), m_metadataCount(0),
          m_alignment(32), m_metadataOnly(false),
          m_tensorDataStart(nullptr), m_tensorDataBaseOffset(0) {}

    GgufLoader::~GgufLoader() { Unload(); }

    void GgufLoader::Unload() {
        m_metadata.clear();
        m_tensors.clear();
        m_fileBuffer.clear();
        m_tensorDataStart       = nullptr;
        m_tensorDataBaseOffset  = 0;
        m_metadataOnly          = false;
    }

    // -----------------------------------------------------------------------
    //  ComputeDataSize — element count -> byte size for each GGML quant type
    // -----------------------------------------------------------------------
    uint64_t GgufLoader::ComputeDataSize(GgmlType type, const std::vector<uint64_t>& shape) {
        uint64_t elements = 1;
        for (auto d : shape) elements *= d;
        switch (type) {
            case GgmlType::F32:  return elements * 4;
            case GgmlType::F16:  return elements * 2;
            case GgmlType::I8:   return elements;
            case GgmlType::I16:  return elements * 2;
            case GgmlType::I32:  return elements * 4;
            case GgmlType::Q4_0: return (elements / 32) * 18;
            case GgmlType::Q4_1: return (elements / 32) * 20;
            case GgmlType::Q5_0: return (elements / 32) * 22;
            case GgmlType::Q5_1: return (elements / 32) * 24;
            case GgmlType::Q8_0: return (elements / 32) * 34;
            case GgmlType::Q8_1: return (elements / 32) * 40;
            case GgmlType::Q2_K: return elements / 4;
            case GgmlType::Q3_K: return (elements * 3) / 8;
            case GgmlType::Q4_K: return elements / 2;
            case GgmlType::Q5_K: return (elements * 5) / 8;
            case GgmlType::Q6_K: return (elements * 6) / 8;
            case GgmlType::Q8_K: return elements;
            default:             return elements * 2;
        }
    }

    // -----------------------------------------------------------------------
    //  ParseAll — stream-based parser, works for both full and metadata-only
    //  metadataOnly=true  : reads up to tensor-data section, then stops
    //  metadataOnly=false : reads entire file into m_fileBuffer first
    // -----------------------------------------------------------------------
    bool GgufLoader::ParseAll(std::ifstream& f, bool metadataOnly) {
        // Helper lambdas that operate on the ifstream directly
        auto readBytes = [&](void* dest, size_t size) -> bool {
            f.read(reinterpret_cast<char*>(dest), size);
            return f.good();
        };

        auto readStr = [&]() -> std::string {
            uint64_t len = 0;
            if (m_version == 1) {
                uint32_t l32 = 0;
                if (!readBytes(&l32, 4)) return "";
                len = l32;
            } else {
                if (!readBytes(&len, 8)) return "";
            }
            if (len == 0) return "";
            std::string s(len, '\0');
            readBytes(&s[0], len);
            return s;
        };

        std::function<GgufValue(GgufType)> readVal = [&](GgufType type) -> GgufValue {
            GgufValue val;
            val.Type = type;
            switch (type) {
                case GgufType::UINT8: case GgufType::INT8: case GgufType::BOOL:
                    val.Data.resize(1); readBytes(val.Data.data(), 1); break;
                case GgufType::UINT16: case GgufType::INT16:
                    val.Data.resize(2); readBytes(val.Data.data(), 2); break;
                case GgufType::UINT32: case GgufType::INT32: case GgufType::FLOAT32:
                    val.Data.resize(4); readBytes(val.Data.data(), 4); break;
                case GgufType::UINT64: case GgufType::INT64: case GgufType::FLOAT64:
                    val.Data.resize(8); readBytes(val.Data.data(), 8); break;
                case GgufType::STRING:
                    val.StringValue = readStr(); break;
                case GgufType::ARRAY: {
                    uint32_t arrTypeVal = 0;
                    readBytes(&arrTypeVal, 4);
                    val.ArrayType = static_cast<GgufType>(arrTypeVal);
                    uint64_t arrLen = 0;
                    if (m_version == 1) { uint32_t l = 0; readBytes(&l, 4); arrLen = l; }
                    else readBytes(&arrLen, 8);
                    val.ArrayLength = arrLen;
                    val.ArrayValues.reserve(arrLen);
                    for (uint64_t i = 0; i < arrLen; ++i)
                        val.ArrayValues.push_back(readVal(val.ArrayType));
                    break;
                }
                default: break;
            }
            return val;
        };

        // --- Header ---
        char magic[4];
        if (!readBytes(magic, 4)) return false;
        if (magic[0]!='G'||magic[1]!='G'||magic[2]!='U'||magic[3]!='F') {
            std::cerr << "[GgufLoader] Bad magic." << std::endl; return false;
        }
        if (!readBytes(&m_version, 4)) return false;
        if (m_version == 1) {
            uint32_t tc, mc;
            readBytes(&tc, 4); readBytes(&mc, 4);
            m_tensorCount = tc; m_metadataCount = mc;
        } else {
            readBytes(&m_tensorCount, 8);
            readBytes(&m_metadataCount, 8);
        }

        // --- Metadata ---
        for (uint64_t i = 0; i < m_metadataCount; ++i) {
            std::string key = readStr();
            uint32_t typeVal = 0;
            if (!readBytes(&typeVal, 4)) return false;
            GgufType valType = static_cast<GgufType>(typeVal);
            GgufValue val = readVal(valType);
            m_metadata[key] = val;
            if (key == "general.alignment") {
                if (val.Type == GgufType::UINT32)
                    m_alignment = *reinterpret_cast<const uint32_t*>(val.Data.data());
                else if (val.Type == GgufType::UINT64)
                    m_alignment = *reinterpret_cast<const uint64_t*>(val.Data.data());
            }
        }

        // --- Tensor descriptors ---
        for (uint64_t i = 0; i < m_tensorCount; ++i) {
            GgufTensor tensor;
            tensor.Name = readStr();

            uint32_t dimsCount = 0;
            if (!readBytes(&dimsCount, 4)) return false;
            tensor.Shape.resize(dimsCount);
            for (uint32_t d = 0; d < dimsCount; ++d) {
                uint64_t dim = 0;
                if (m_version == 1) { uint32_t d32 = 0; readBytes(&d32, 4); dim = d32; }
                else readBytes(&dim, 8);
                tensor.Shape[d] = dim;
            }
            uint32_t typeVal = 0;
            readBytes(&typeVal, 4);
            tensor.Type = static_cast<GgmlType>(typeVal);
            readBytes(&tensor.FileOffset, 8);
            tensor.DataSize = ComputeDataSize(tensor.Type, tensor.Shape);
            tensor.DataPtr  = nullptr; // set later for full-load
            m_tensors[tensor.Name] = tensor;
        }

        // --- Align to tensor data section ---
        uint64_t curPos = (uint64_t)f.tellg();
        uint64_t aligned = (curPos + m_alignment - 1) & ~(m_alignment - 1);

        if (metadataOnly) {
            // Just record the base offset; don't read data
            m_tensorDataBaseOffset = aligned;
            return true;
        }

        // Full load: data is already in m_fileBuffer from caller; set DataPtr
        if (aligned <= m_fileBuffer.size()) {
            m_tensorDataStart = m_fileBuffer.data() + aligned;
        } else {
            std::cerr << "[GgufLoader] File truncated at data section." << std::endl;
            return false;
        }
        for (auto& [name, t] : m_tensors)
            t.DataPtr = m_tensorDataStart + t.FileOffset;

        return true;
    }

    // -----------------------------------------------------------------------
    //  LoadFile — reads entire file into buffer, then parses
    //  Suitable for models that fit in RAM (< available RAM)
    // -----------------------------------------------------------------------
    bool GgufLoader::LoadFile(const std::string& filepath) {
        Unload();
        m_filepath = filepath;
        m_metadataOnly = false;

        std::ifstream f(filepath, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            std::cerr << "[GgufLoader] Cannot open: " << filepath << std::endl;
            return false;
        }
        size_t fileSize = (size_t)f.tellg();
        f.seekg(0, std::ios::beg);

        m_fileBuffer.resize(fileSize);
        if (!f.read(reinterpret_cast<char*>(m_fileBuffer.data()), fileSize)) {
            std::cerr << "[GgufLoader] Read failed: " << filepath << std::endl;
            return false;
        }
        f.close();

        // Re-open as stream for the parser (uses the same data via seekg=0 trick,
        // but parseAll works on ifstream directly to keep one code path)
        std::ifstream f2(filepath, std::ios::binary);
        if (!ParseAll(f2, false)) return false;

        std::cout << "[GgufLoader] Full load: " << m_tensorCount << " tensors, "
                  << (fileSize / (1024*1024)) << " MB" << std::endl;
        return true;
    }

    // -----------------------------------------------------------------------
    //  LoadMetadataOnly — parses header + tensor descriptors only.
    //  Tensor data is NOT buffered. DataPtr == nullptr for all tensors.
    //  Use for 100GB+ models: let DirectStorage stream experts on demand.
    // -----------------------------------------------------------------------
    bool GgufLoader::LoadMetadataOnly(const std::string& filepath) {
        Unload();
        m_filepath = filepath;
        m_metadataOnly = true;

        std::ifstream f(filepath, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "[GgufLoader] Cannot open: " << filepath << std::endl;
            return false;
        }

        if (!ParseAll(f, true)) return false;

        std::cout << "[GgufLoader] Metadata-only: " << m_tensorCount
                  << " tensors parsed. Data base offset=" << m_tensorDataBaseOffset
                  << ". Tensor data NOT buffered (use DirectStorage to stream)." << std::endl;
        return true;
    }

    // -----------------------------------------------------------------------
    //  Metadata accessors
    // -----------------------------------------------------------------------
    bool GgufLoader::HasMetadata(const std::string& key) const {
        return m_metadata.count(key) > 0;
    }
    std::string GgufLoader::GetMetadataString(const std::string& key, const std::string& def) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() && it->second.Type == GgufType::STRING)
            return it->second.StringValue;
        return def;
    }
    uint32_t GgufLoader::GetMetadataUint32(const std::string& key, uint32_t def) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() &&
            (it->second.Type==GgufType::UINT32||it->second.Type==GgufType::INT32))
            return *reinterpret_cast<const uint32_t*>(it->second.Data.data());
        return def;
    }
    uint64_t GgufLoader::GetMetadataUint64(const std::string& key, uint64_t def) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() &&
            (it->second.Type==GgufType::UINT64||it->second.Type==GgufType::INT64))
            return *reinterpret_cast<const uint64_t*>(it->second.Data.data());
        return def;
    }
    float GgufLoader::GetMetadataFloat(const std::string& key, float def) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() && it->second.Type == GgufType::FLOAT32)
            return *reinterpret_cast<const float*>(it->second.Data.data());
        return def;
    }
    const GgufTensor* GgufLoader::GetTensor(const std::string& name) const {
        auto it = m_tensors.find(name);
        return (it != m_tensors.end()) ? &it->second : nullptr;
    }
}
