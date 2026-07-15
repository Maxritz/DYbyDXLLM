// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>

namespace DirectLLM {

    enum class GgmlType : uint32_t {
        F32 = 0, F16 = 1,
        Q4_0 = 2, Q4_1 = 3,
        Q5_0 = 6, Q5_1 = 7,
        Q8_0 = 8, Q8_1 = 9,
        Q2_K = 10, Q3_K = 11, Q4_K = 12,
        Q5_K = 13, Q6_K = 14, Q8_K = 15,
        I8 = 16, I16 = 17, I32 = 18,
        Count
    };

    enum class GgufType : uint32_t {
        UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3,
        UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7,
        STRING = 8, ARRAY = 9, UINT64 = 10,
        INT64 = 11, FLOAT64 = 12
    };

    struct GgufValue {
        GgufType Type;
        std::vector<uint8_t> Data;
        std::string StringValue;
        GgufType ArrayType;
        uint64_t ArrayLength;
        std::vector<GgufValue> ArrayValues;
    };

    struct GgufTensor {
        std::string Name;
        GgmlType    Type;
        std::vector<uint64_t> Shape;

        // Absolute offset from start-of-data-section.
        // absoluteFileOffset = GetTensorDataBaseOffset() + FileOffset
        uint64_t FileOffset  = 0;
        uint64_t DataSize    = 0;

        // DataPtr is valid only in full-load mode.
        // In metadata-only mode DataPtr == nullptr; use FileOffset + DirectStorage.
        uint8_t* DataPtr = nullptr;
    };

    class GgufLoader {
    public:
        GgufLoader();
        ~GgufLoader();

        // Full load: reads entire file into m_fileBuffer.
        // Use for models that fit in RAM.
        bool LoadFile(const std::string& filepath);

        // Metadata-only load: parses header + tensor descriptors only.
        // Tensor data is NOT read into RAM (DataPtr == nullptr).
        // Use for 100GB+ MoE models; stream tensors on-demand via DirectStorage.
        bool LoadMetadataOnly(const std::string& filepath);

        void Unload();

        // Metadata accessors
        bool        HasMetadata(const std::string& key) const;
        std::string GetMetadataString(const std::string& key, const std::string& defaultVal = "") const;
        uint32_t    GetMetadataUint32(const std::string& key, uint32_t defaultVal = 0) const;
        uint64_t    GetMetadataUint64(const std::string& key, uint64_t defaultVal = 0) const;
        float       GetMetadataFloat (const std::string& key, float    defaultVal = 0.0f) const;

        // Tensor accessors
        const GgufTensor* GetTensor(const std::string& name) const;
        const std::unordered_map<std::string, GgufTensor>& GetTensors() const { return m_tensors; }
        const std::unordered_map<std::string, GgufValue>&  GetAllMetadata() const { return m_metadata; }

        // Version / counts
        uint32_t GetVersion()       const { return m_version; }
        uint64_t GetTensorCount()   const { return m_tensorCount; }
        uint64_t GetMetadataCount() const { return m_metadataCount; }
        uint64_t GetAlignment()     const { return m_alignment; }

        // DirectStorage helpers
        // Absolute byte offset of the tensor-data section from start of file.
        // Use: absoluteOffset = GetTensorDataBaseOffset() + tensor.FileOffset
        uint64_t GetTensorDataBaseOffset() const {
            if (m_metadataOnly) return m_tensorDataBaseOffset;
            return (m_tensorDataStart && !m_fileBuffer.empty())
                   ? (uint64_t)(m_tensorDataStart - m_fileBuffer.data())
                   : 0ULL;
        }
        const std::string& GetFilePath() const { return m_filepath; }
        bool IsMetadataOnly() const { return m_metadataOnly; }

    private:
        std::string m_filepath;
        uint32_t m_version       = 0;
        uint64_t m_tensorCount   = 0;
        uint64_t m_metadataCount = 0;
        uint64_t m_alignment     = 32;
        bool     m_metadataOnly  = false;

        std::unordered_map<std::string, GgufValue>  m_metadata;
        std::unordered_map<std::string, GgufTensor> m_tensors;

        // Full-load mode
        std::vector<uint8_t> m_fileBuffer;
        uint8_t*             m_tensorDataStart = nullptr;

        // Metadata-only mode: store offset directly
        uint64_t m_tensorDataBaseOffset = 0;

        // Shared parsing helpers (work on ifstream, no full buffer needed)
        bool ParseAll(std::ifstream& f, bool metadataOnly);
        static uint64_t ComputeDataSize(GgmlType type, const std::vector<uint64_t>& shape);
    };
}
