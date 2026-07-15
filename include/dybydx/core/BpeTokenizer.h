// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace DirectLLM {

    class GgufLoader; // forward declaration

    class BpeTokenizer {
    public:
        BpeTokenizer();
        ~BpeTokenizer();

        bool LoadVocabulary(const std::string& vocabPath);
        bool LoadFromGGUF(const GgufLoader& loader);

        // Wrap a raw user prompt in the model's instruct template.
        // Call BEFORE Encode(). arch = GGUF general.architecture string.
        std::string ApplyChatTemplate(const std::string& userPrompt,
                                       const std::string& arch) const;

        std::vector<int32_t> Encode(const std::string& text);
        std::string Decode(const std::vector<int32_t>& ids);

        int32_t GetBosTokenId() const { return m_bosId; }
        int32_t GetEosTokenId() const { return m_eosId; }
        int32_t GetPadTokenId() const { return m_padId; }

        // True if id is any known EOS / end-of-turn token for this model
        bool IsEosToken(int32_t id) const;

    private:
        int32_t m_bosId = 1;
        int32_t m_eosId = 2;
        int32_t m_padId = 0;
        std::vector<int32_t> m_eosIds; // all EOS-equivalent token IDs from GGUF
        std::unordered_map<std::string, int32_t> m_encoder;
        std::unordered_map<int32_t, std::string> m_decoder;
        std::unordered_map<std::string, int64_t> m_bpeRanks;
    };
}
