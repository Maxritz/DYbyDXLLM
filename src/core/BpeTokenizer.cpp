#include "dybydx/core/BpeTokenizer.h"
#include "dybydx/core/GgufLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <cstdint>

namespace DirectLLM {

    // ---------------------------------------------------------------------
    //  GPT-2 byte <-> unicode mapping (bytes_to_unicode from the GPT-2 repo).
    //  Vocab strings in byte-level BPE models (llama3/qwen/minicpm) are stored
    //  in this encoding: printable bytes map to themselves, everything else
    //  maps to codepoints 256+n. Space becomes U+0120 (Ġ), newline U+010A (Ċ).
    // ---------------------------------------------------------------------
    static const uint16_t* Gpt2ByteToCp() {
        static uint16_t table[256];
        static bool init = false;
        if (!init) {
            int n = 0;
            for (int b = 0; b < 256; ++b) {
                bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
                table[b] = printable ? (uint16_t)b : (uint16_t)(256 + n++);
            }
            init = true;
        }
        return table;
    }

    static void AppendUtf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) s += (char)cp;
        else if (cp < 0x800) {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        } else {
            s += (char)(0xE0 | (cp >> 12));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        }
    }

    std::string BpeTokenizer::ByteEncode(const std::string& raw) const {
        const uint16_t* table = Gpt2ByteToCp();
        std::string out;
        out.reserve(raw.size() * 2);
        for (unsigned char c : raw) AppendUtf8(out, table[c]);
        return out;
    }

    std::string BpeTokenizer::ByteDecode(const std::string& mapped) const {
        // Reverse map: codepoint -> byte
        static std::unordered_map<uint32_t, uint8_t> rev;
        if (rev.empty()) {
            const uint16_t* table = Gpt2ByteToCp();
            for (int b = 0; b < 256; ++b) rev[table[b]] = (uint8_t)b;
        }
        std::string out;
        size_t i = 0;
        while (i < mapped.size()) {
            unsigned char c = (unsigned char)mapped[i];
            uint32_t cp; size_t len;
            if (c < 0x80)            { cp = c; len = 1; }
            else if ((c >> 5) == 6)  { cp = c & 0x1F; len = 2; }
            else if ((c >> 4) == 14) { cp = c & 0x0F; len = 3; }
            else                     { cp = c & 0x07; len = 4; }
            if (i + len > mapped.size()) break;
            for (size_t j = 1; j < len; ++j) cp = (cp << 6) | ((unsigned char)mapped[i + j] & 0x3F);
            auto it = rev.find(cp);
            if (it != rev.end()) out += (char)it->second;
            else out.append(mapped, i, len); // unknown codepoint: pass through
            i += len;
        }
        return out;
    }

    BpeTokenizer::BpeTokenizer() {
        m_encoder["<pad>"] = 0;  m_decoder[0] = "<pad>";
        m_encoder["<s>"]   = 1;  m_decoder[1] = "<s>";
        m_encoder["</s>"]  = 2;  m_decoder[2] = "</s>";
        m_encoder["<unk>"] = 3;  m_decoder[3] = "<unk>";
    }

    BpeTokenizer::~BpeTokenizer() {}

    bool BpeTokenizer::LoadFromGGUF(const GgufLoader& loader) {
        // 1. Load the token table
        for (auto& [key, val] : loader.GetAllMetadata()) {
            if (key == "tokenizer.ggml.tokens" && val.Type == GgufType::ARRAY) {
                for (size_t i = 0; i < val.ArrayValues.size(); i++) {
                    auto& v = val.ArrayValues[i];
                    if (v.Type == GgufType::STRING && !v.StringValue.empty()) {
                        int32_t id = (int32_t)(i);
                        m_encoder[v.StringValue] = id;
                        m_decoder[id] = v.StringValue;
                    }
                }
            }
            if (key == "tokenizer.ggml.scores" && val.Type == GgufType::ARRAY) {
                for (size_t i = 0; i < val.ArrayValues.size(); i++) {
                    auto& v = val.ArrayValues[i];
                    if (v.Type == GgufType::FLOAT32 && v.Data.size() == 4) {
                        float score = *reinterpret_cast<const float*>(v.Data.data());
                        m_bpeRanks[std::to_string(i)] = (int64_t)(score * 1000);
                    }
                }
            }
        }

        // 1b. Tokenizer flavor + chat template + add_bos flag
        std::string tokModel = loader.GetMetadataString("tokenizer.ggml.model", "");
        m_byteLevel = (tokModel == "gpt2");
        m_spm       = (tokModel == "llama");
        m_chatTemplate = loader.GetMetadataString("tokenizer.chat_template", "");
        {
            auto& all = loader.GetAllMetadata();
            auto it = all.find("tokenizer.ggml.add_bos_token");
            if (it != all.end() && it->second.Type == GgufType::BOOL && !it->second.Data.empty())
                m_addBos = it->second.Data[0] != 0;
        }
        std::cout << "[Tokenizer] model=" << (tokModel.empty() ? "(none)" : tokModel)
                  << " byteLevel=" << m_byteLevel << " spm=" << m_spm
                  << " addBos=" << m_addBos
                  << " chatTemplate=" << (m_chatTemplate.empty() ? "no" : "yes") << std::endl;

        // 2. Read BOS / EOS token IDs from GGUF metadata
        if (loader.HasMetadata("tokenizer.ggml.bos_token_id")) {
            m_bosId = (int32_t)loader.GetMetadataUint32("tokenizer.ggml.bos_token_id");
        }
        if (loader.HasMetadata("tokenizer.ggml.eos_token_id")) {
            m_eosId = (int32_t)loader.GetMetadataUint32("tokenizer.ggml.eos_token_id");
        }
        // Collect all EOS-equivalent ids (some models have multiple)
        m_eosIds.clear();
        m_eosIds.push_back(m_eosId);
        // Common end-of-turn / end-of-text tokens to also stop on
        static const std::vector<std::string> kEosStrings = {
            "</s>", "<|endoftext|>", "<|end|>", "<end_of_turn>",
            "<|im_end|>", "<eot_id>", "<|eom_id|>", "<|end_of_sentence|>"
        };
        for (auto& s : kEosStrings) {
            auto it = m_encoder.find(s);
            if (it != m_encoder.end()) {
                int32_t id = it->second;
                if (std::find(m_eosIds.begin(), m_eosIds.end(), id) == m_eosIds.end())
                    m_eosIds.push_back(id);
            }
        }

        std::cout << "[Tokenizer] Loaded " << m_encoder.size()
                  << " tokens from GGUF metadata."
                  << " BOS=" << m_bosId << " EOS=" << m_eosId << std::endl;
        return m_encoder.size() > 4;
    }

    bool BpeTokenizer::LoadVocabulary(const std::string& vocabPath) {
        std::ifstream file(vocabPath);
        if (!file.is_open()) {
            std::cout << "[Tokenizer] No vocab file at " << vocabPath << ", using fallback." << std::endl;
            return false;
        }
        std::string line;
        while (std::getline(file, line)) {
            auto pos = line.find(' ');
            if (pos != std::string::npos) {
                std::string word = line.substr(0, pos);
                int32_t id = std::stoi(line.substr(pos + 1));
                m_encoder[word] = id;
                m_decoder[id] = word;
            }
        }
        std::cout << "[Tokenizer] Loaded " << m_encoder.size() << " tokens from " << vocabPath << std::endl;
        return true;
    }

    // -------------------------------------------------------------------------
    //  ApplyChatTemplate
    //  Wraps the raw user prompt in the correct instruct template for each arch.
    //  The result is the full string to pass to Encode().
    // -------------------------------------------------------------------------
    std::string BpeTokenizer::ApplyChatTemplate(const std::string& userPrompt,
                                                  const std::string& arch) const {
        // Prefer the model's own chat template (tokenizer.chat_template) to
        // pick the format; base models without one get the raw prompt.
        if (m_chatTemplate.empty())
            return userPrompt;
        if (m_chatTemplate.find("<|im_start|>") != std::string::npos) {
            return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                   "<|im_start|>user\n" + userPrompt + "<|im_end|>\n"
                   "<|im_start|>assistant\n";
        }
        if (m_chatTemplate.find("start_of_turn") != std::string::npos) {
            return "<bos><start_of_turn>user\n" + userPrompt + "<end_of_turn>\n"
                   "<start_of_turn>model\n";
        }
        if (m_chatTemplate.find("start_header_id") != std::string::npos) {
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                   + userPrompt +
                   "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        }
        if (m_chatTemplate.find("<|user|>") != std::string::npos) {
            return "<|user|>\n" + userPrompt + "<|end|>\n<|assistant|>\n";
        }
        if (m_chatTemplate.find("[INST]") != std::string::npos) {
            return "[INST] " + userPrompt + " [/INST]";
        }
        // Unrecognized template: fall back to arch-based guess below.
        // qwen2 / qwen25 / qwen35 (MiniCPM uses qwen35 arch)
        if (arch == "qwen2" || arch == "qwen25" || arch == "qwen35" ||
            arch.find("qwen") != std::string::npos) {
            return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                   "<|im_start|>user\n" + userPrompt + "<|im_end|>\n"
                   "<|im_start|>assistant\n";
        }
        // gemma / gemma2 / gemma4
        if (arch.find("gemma") != std::string::npos) {
            return "<bos><start_of_turn>user\n" + userPrompt + "<end_of_turn>\n"
                   "<start_of_turn>model\n";
        }
        // phi3 / phi
        if (arch.find("phi") != std::string::npos) {
            return "<|user|>\n" + userPrompt + "<|end|>\n<|assistant|>\n";
        }
        // llama3 / llama
        if (arch.find("llama") != std::string::npos) {
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                   + userPrompt +
                   "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        }
        // deepseek / mistral: simple [INST] format
        return "[INST] " + userPrompt + " [/INST]";
    }

    bool BpeTokenizer::IsEosToken(int32_t id) const {
        return std::find(m_eosIds.begin(), m_eosIds.end(), id) != m_eosIds.end();
    }

    std::vector<int32_t> BpeTokenizer::Encode(const std::string& text) {
        std::vector<int32_t> tokens;

        // Prepare the text in the vocab's own encoding:
        //  byte-level (gpt2): map raw bytes -> GPT-2 unicode. ASCII printables
        //    map to themselves, so literal special tokens like <|im_start|>
        //    still match; spaces become Ġ so "Ġthe"-style tokens finally match.
        //  SPM (llama): spaces become ▁ (U+2581) with a leading space prefix.
        std::string prepared;
        if (m_byteLevel) {
            prepared = ByteEncode(text);
        } else if (m_spm) {
            std::string t = " " + text;
            for (size_t i = 0; i < t.size(); ++i) {
                if (t[i] == ' ') prepared += "\xE2\x96\x81";
                else prepared += t[i];
            }
        } else {
            prepared = text;
        }

        // BOS: honor tokenizer.ggml.add_bos_token unless the template already
        // starts with the BOS token's literal string (avoids double BOS).
        if (m_addBos) {
            auto bosStr = m_decoder.find(m_bosId);
            bool alreadyThere = bosStr != m_decoder.end() && !bosStr->second.empty() &&
                                text.rfind(bosStr->second, 0) == 0;
            if (!alreadyThere) tokens.push_back(m_bosId);
        }

        // Greedy longest-match (capped: no vocab token is longer than ~64 bytes)
        size_t i = 0;
        const size_t maxTokLen = 128;
        while (i < prepared.length()) {
            std::string best;
            size_t bestLen = 0;
            size_t maxLen = std::min(maxTokLen, prepared.length() - i);
            for (size_t len = maxLen; len >= 1; len--) {
                std::string sub = prepared.substr(i, len);
                if (m_encoder.find(sub) != m_encoder.end()) {
                    best = sub; bestLen = len; break;
                }
            }
            if (bestLen > 0) {
                tokens.push_back(m_encoder.at(best));
                i += bestLen;
            } else {
                // SPM fallback: byte tokens are stored as "<0xNN>"
                if (m_spm) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned char)prepared[i]);
                    auto it = m_encoder.find(buf);
                    if (it != m_encoder.end()) { tokens.push_back(it->second); i++; continue; }
                }
                auto it = m_encoder.find(std::string(1, prepared[i]));
                if (it != m_encoder.end()) tokens.push_back(it->second);
                else if (m_encoder.count("<unk>")) tokens.push_back(m_encoder.at("<unk>"));
                i++;
            }
        }
        return tokens;
    }

    std::string BpeTokenizer::Decode(const std::vector<int32_t>& ids) {
        std::string text;
        for (int32_t id : ids) {
            if (IsEosToken(id)) break;
            if (id == m_bosId || id == m_padId) continue;
            auto it = m_decoder.find(id);
            if (it == m_decoder.end()) continue;
            const std::string& tok = it->second;

            // Skip control/special tokens in output (e.g. <|im_start|>)
            if (tok.size() > 2 && tok.front() == '<' && tok.back() == '>' &&
                !(m_spm && tok.rfind("<0x", 0) == 0))
                continue;

            if (m_byteLevel) {
                // Full GPT-2 byte decode (handles Ġ, Ċ, and all mapped bytes)
                text += ByteDecode(tok);
            } else if (m_spm) {
                if (tok.rfind("<0x", 0) == 0 && tok.size() == 6) {
                    // SPM raw byte token "<0xNN>"
                    text += (char)std::stoi(tok.substr(3, 2), nullptr, 16);
                } else {
                    // Replace ▁ (0xE2 0x96 0x81) with space
                    for (size_t k = 0; k < tok.size(); ) {
                        if (k + 2 < tok.size() &&
                            (unsigned char)tok[k] == 0xE2 &&
                            (unsigned char)tok[k+1] == 0x96 &&
                            (unsigned char)tok[k+2] == 0x81) {
                            text += ' '; k += 3;
                        } else { text += tok[k]; k++; }
                    }
                }
            } else {
                text += tok;
            }
        }
        return text;
    }
}
