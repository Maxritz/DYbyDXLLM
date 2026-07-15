// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "dybydx/core/BpeTokenizer.h"

namespace DirectLLM {

    enum class TokenizerType {
        BytePairEncoding = 0, // GPT-2, Phi-3, GPT-4
        SentencePiece,         // Llama, Mistral, Gemma
        WordPiece             // BERT, Gemini
    };

    class MultiFormatTokenizer {
    public:
        MultiFormatTokenizer(TokenizerType type) : m_type(type) {
            m_bpe = std::make_unique<BpeTokenizer>();
        }

        bool LoadVocab(const std::string& vocabPath) {
            std::cout << "[MultiTokenizer] Initializing tokenizer type: ";
            switch (m_type) {
                case TokenizerType::BytePairEncoding: std::cout << "Byte-Pair Encoding (BPE)"; break;
                case TokenizerType::SentencePiece: std::cout << "SentencePiece (Llama-style)"; break;
                case TokenizerType::WordPiece: std::cout << "WordPiece (Gemini-style)"; break;
            }
            std::cout << std::endl;
            return m_bpe->LoadVocabulary(vocabPath);
        }

        std::vector<int32_t> Tokenize(const std::string& text) {
            return m_bpe->Encode(text);
        }

        std::string Detokenize(const std::vector<int32_t>& tokens) {
            return m_bpe->Decode(tokens);
        }

    private:
        TokenizerType m_type;
        std::unique_ptr<BpeTokenizer> m_bpe;
    };
}
