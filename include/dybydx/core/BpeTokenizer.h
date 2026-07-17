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

        // Tokenizer flavor from tokenizer.ggml.model:
        //  "gpt2"  -> byte-level BPE (llama3, qwen, minicpm): vocab strings are
        //             GPT-2 byte-encoded; raw text must be mapped before greedy
        //             matching and mapped back on decode.
        //  "llama" -> SentencePiece: spaces are U+2581 (▁).
        bool m_byteLevel = false;
        bool m_spm = false;
        bool m_addBos = true;
        std::string m_chatTemplate; // tokenizer.chat_template if present

        std::string ByteEncode(const std::string& raw) const;   // raw bytes -> gpt2 unicode
        std::string ByteDecode(const std::string& mapped) const; // gpt2 unicode -> raw bytes
    };
}
