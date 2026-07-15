// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>

namespace DirectLLM {

    enum class GgmlType : uint32_t {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        I8 = 16,
        I16 = 17,
        I32 = 18,
        Count
    };

    enum class GgufType : uint32_t {
        UINT8 = 0,
        INT8 = 1,
        UINT16 = 2,
        INT16 = 3,
        UINT32 = 4,
        INT32 = 5,
        FLOAT32 = 6,
        BOOL = 7,
        STRING = 8,
        ARRAY = 9,
        UINT64 = 10,
        INT64 = 11,
        FLOAT64 = 12
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
        GgmlType Type;
        std::vector<uint64_t> Shape;
        uint64_t FileOffset;
        uint64_t DataSize;
        uint8_t* DataPtr;
    };

    class GgufLoader {
    public:
        GgufLoader();
        ~GgufLoader();

        bool LoadFile(const std::string& filepath);
        void Unload();

        bool HasMetadata(const std::string& key) const;
        std::string GetMetadataString(const std::string& key, const std::string& defaultVal = "") const;
        uint32_t GetMetadataUint32(const std::string& key, uint32_t defaultVal = 0) const;
        uint64_t GetMetadataUint64(const std::string& key, uint64_t defaultVal = 0) const;
        float GetMetadataFloat(const std::string& key, float defaultVal = 0.0f) const;

        const GgufTensor* GetTensor(const std::string& name) const;
        const std::unordered_map<std::string, GgufTensor>& GetTensors() const { return m_tensors; }
        const std::unordered_map<std::string, GgufValue>& GetAllMetadata() const { return m_metadata; }

        uint32_t GetVersion() const { return m_version; }
        uint64_t GetTensorCount() const { return m_tensorCount; }
        uint64_t GetMetadataCount() const { return m_metadataCount; }
        uint64_t GetAlignment() const { return m_alignment; }

    private:
        std::string m_filepath;
        uint32_t m_version = 0;
        uint64_t m_tensorCount = 0;
        uint64_t m_metadataCount = 0;
        uint64_t m_alignment = 32;

        std::unordered_map<std::string, GgufValue> m_metadata;
        std::unordered_map<std::string, GgufTensor> m_tensors;

        std::vector<uint8_t> m_fileBuffer;
        uint8_t* m_tensorDataStart = nullptr;

        bool ParseHeader(std::ifstream& file);
        bool ParseMetadata(std::ifstream& file);
        bool ParseTensorInfos(std::ifstream& file);
        GgufValue ReadValue(std::ifstream& file, GgufType type);
        std::string ReadString(std::ifstream& file);
    };
}
