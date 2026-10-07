#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/tokenizer/nfc.h"
#include "core/tokenizer/unicode.h"
#include "core/tokenizer/unicode_tables.h"
#include "support/test_data.h"

using namespace bllm::tokenizer;

namespace {

std::string nfc(std::string_view text) {
    std::string out;
    REQUIRE(to_nfc(text, out));
    return out;
}

// "1E0A 0323" as UTF-8.
std::string from_hex(std::string_view field) {
    std::string out;
    std::istringstream points{std::string(field)};
    for (std::string hex; points >> hex;) append_utf8(static_cast<char32_t>(std::stoul(hex, nullptr, 16)), out);
    return out;
}

std::vector<std::string_view> split(std::string_view line, char separator) {
    std::vector<std::string_view> fields;
    for (std::size_t start = 0;;) {
        const std::size_t end = line.find(separator, start);
        fields.push_back(line.substr(start, end - start));
        if (end == std::string_view::npos) return fields;
        start = end + 1;
    }
}

}  // namespace

TEST_CASE("NFC composes, reorders and leaves composed text alone") {
    CHECK(nfc("cafe\xCC\x81") == "caf\xC3\xA9");                  // e + acute -> é
    CHECK(nfc("caf\xC3\xA9") == "caf\xC3\xA9");                   // already composed
    CHECK(nfc("") == "");
    CHECK(nfc("plain ASCII stays as it is") == "plain ASCII stays as it is");
    // Below (220) sorts before above (230), then each composes in turn:
    // q has no composite with either, so both stay, ordered.
    CHECK(nfc("q\xCC\x87\xCC\xA3") == "q\xCC\xA3\xCC\x87");
    // A Hangul leading consonant, vowel and trailing consonant compose.
    CHECK(nfc("\xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB") == "\xED\x95\x9C");   // 한
}

TEST_CASE("NFC shrinks a character's canonical equivalents at most 7 bytes to 2") {
    // The generator's pre-encode bound (runtime/generator.h) rests on this:
    // each character NFC writes stands for at most its canonical
    // equivalents' bytes. A character's longest equivalent spells its full
    // decomposition piece by piece, each piece the longest character whose
    // full decomposition it is; computed here, over the tables, for every
    // character NFC leaves alone.
    const auto pool = unicode_tables::decomposition_pool();
    std::map<std::u32string, std::size_t> longest_spelling;   // a full decomposition: its longest character's bytes
    const auto bytes_of = [](char32_t c) {
        std::string s;
        append_utf8(c, s);
        return s.size();
    };
    for (const auto& d : unicode_tables::decompositions()) {
        const std::u32string full(pool.data() + d.start, d.length);
        auto& best = longest_spelling[full];
        best = std::max(best, bytes_of(d.code_point));
    }
    std::size_t most = 0, least = 1;   // the greatest ratio, most / least
    char32_t at = 0;
    for (const auto& d : unicode_tables::decompositions()) {
        std::string self;
        append_utf8(d.code_point, self);
        if (nfc(self) != self) continue;   // NFC never writes it
        const std::u32string full(pool.data() + d.start, d.length);
        std::vector<std::size_t> spelled(full.size() + 1, 0);   // longest spelling of full's first i
        for (std::size_t i = 1; i <= full.size(); ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                if (j > 0 && spelled[j] == 0) continue;
                const std::u32string piece = full.substr(j, i - j);
                std::size_t piece_bytes = piece.size() == 1 ? bytes_of(piece[0]) : 0;
                if (const auto it = longest_spelling.find(piece); it != longest_spelling.end()) {
                    piece_bytes = std::max(piece_bytes, it->second);
                }
                if (piece_bytes > 0) spelled[i] = std::max(spelled[i], spelled[j] + piece_bytes);
            }
        }
        if (spelled.back() * least > most * self.size()) {
            most = spelled.back();
            least = self.size();
            at = d.code_point;
        }
    }
    CHECK(most == 7);
    CHECK(least == 2);
    CHECK(at == U'ΐ');
    // The equivalent that reaches it: U+1FBE, whose decomposition is U+03B9,
    // then U+0308 and U+0301.
    CHECK(nfc("\xE1\xBE\xBE\xCC\x88\xCC\x81") == "\xCE\x90");
    // A Hangul syllable, arithmetic and not in the tables, is at most its
    // three 3-byte jamo: 9 bytes to 3, under 7 to 2.
    CHECK(nfc("\xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB").size() == 3);
}

TEST_CASE("canonical order keeps marks of the same class in the order they came") {
    // 24 marks, alternating above (class 230) and below (class 220), each one
    // distinct. Canonical order puts every below mark first and every above
    // mark after, each group in its original order. Long enough that an
    // unstable sort, which a short run would hide, reorders equal marks.
    constexpr char32_t kAbove[] = {0x300, 0x301, 0x302, 0x303, 0x304, 0x305,
                                   0x306, 0x307, 0x308, 0x309, 0x30A, 0x30B};
    constexpr char32_t kBelow[] = {0x316, 0x317, 0x318, 0x319, 0x31C, 0x31D,
                                   0x31E, 0x31F, 0x320, 0x323, 0x324, 0x325};
    std::string text = "#";   // composes with none of them
    std::string above, below;
    for (std::size_t i = 0; i < 12; ++i) {
        append_utf8(kAbove[i], text);
        append_utf8(kBelow[i], text);
        append_utf8(kAbove[i], above);
        append_utf8(kBelow[i], below);
    }
    CHECK(nfc(text) == "#" + below + above);
}

TEST_CASE("text below U+0300 is its own NFC, as the full algorithm also finds") {
    // The fast path returns such text as it is. Ending it with a character
    // the full path must handle (U+4E00, which composes with nothing) sends
    // every character below U+0300 through decomposition, reordering and
    // composition instead; none may change, nor any neighbour.
    std::string below;
    for (char32_t c = 0; c < 0x0300; ++c) append_utf8(c, below);
    CHECK(nfc(below) == below);
    CHECK(nfc(below + "\xE4\xB8\x80") == below + "\xE4\xB8\x80");
    // And the fast path still refuses what is not UTF-8.
    std::string out;
    CHECK_FALSE(to_nfc("plain \xC0\x80", out));
    CHECK_FALSE(to_nfc("truncated \xC3", out));
}

TEST_CASE("only the spans NFC could change are normalized, the rest copied") {
    // Each text and its NFC from Python's unicodedata (Unicode 16.0). The
    // spans fall at the start, middle and end of plain text, back to back,
    // and around a starter that composes with what follows it.
    struct Case {
        std::string_view name;
        std::string_view text;
        std::string_view nfc;
    };
    constexpr Case kCases[] = {
        {"a mark at the very start", "\xCC\x81" "abc", "\xCC\x81" "abc"},
        {"a mark after its starter, mid text", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxcafe\xCC\x81 and more yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxcaf\xC3\xA9 and more yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"},
        {"two sequences apart", "e\xCC\x81 plain text  plain text  plain text  plain text  plain text a\xCC\x88", "\xC3\xA9 plain text  plain text  plain text  plain text  plain text \xC3\xA4"},
        {"a sequence ending the text", "plain text then o\xCC\x82", "plain text then \xC3\xB4"},
        {"marks out of order after a starter", "q\xCC\x87\xCC\xA3 then q", "q\xCC\xA3\xCC\x87 then q"},
        {"Hangul jamo between ASCII", "x \xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB y", "x \xED\x95\x9C y"},
        {"a CJK run with a mark", "abc \xE4\xB8\x80\xE4\xBA\x8C\xCC\x81 def", "abc \xE4\xB8\x80\xE4\xBA\x8C\xCC\x81 def"},
        {"a starter that composes, then plain", "A\xCC\x8A then Angstrom \xE2\x84\xAB", "\xC3\x85 then Angstrom \xC3\x85"},
    };
    for (const Case& c : kCases) {
        CAPTURE(c.name);
        CHECK(nfc(c.text) == c.nfc);
    }
}

TEST_CASE("NFC refuses ill-formed UTF-8 and leaves the output untouched") {
    std::string out = "unchanged";
    CHECK_FALSE(to_nfc("ab\xC0\x80", out));
    CHECK(out == "unchanged");
}

TEST_CASE("NFC conforms to Unicode's NormalizationTest.txt") {
    const std::string file = bllm::testing::load_test_data("unicode/NormalizationTest-16.0.0.txt");
    std::set<char32_t> listed;   // Part 1's characters
    bool part1 = false;
    std::size_t cases = 0;
    std::istringstream lines{file};
    for (std::string line; std::getline(lines, line);) {
        if (line.empty() || line[0] == '#') continue;
        if (line[0] == '@') {
            part1 = line.starts_with("@Part1");
            continue;
        }
        const auto fields = split(line, ';');
        REQUIRE(fields.size() >= 5);
        const std::string c1 = from_hex(fields[0]), c2 = from_hex(fields[1]), c3 = from_hex(fields[2]),
                          c4 = from_hex(fields[3]), c5 = from_hex(fields[4]);
        // The conformance conditions for NFC, from the file's header.
        CAPTURE(line);
        CHECK(nfc(c1) == c2);
        CHECK(nfc(c2) == c2);
        CHECK(nfc(c3) == c2);
        CHECK(nfc(c4) == c4);
        CHECK(nfc(c5) == c4);
        if (part1) listed.insert(static_cast<char32_t>(std::stoul(std::string(fields[0]), nullptr, 16)));
        ++cases;
    }
    CHECK(cases > 19'000);

    // And every character Part 1 does not list is its own NFC.
    for (char32_t c = 0; c <= 0x10FFFF; ++c) {
        if ((c >= 0xD800 && c <= 0xDFFF) || listed.contains(c)) continue;
        std::string one;
        append_utf8(c, one);
        if (nfc(one) != one) FAIL("U+" << std::hex << static_cast<std::uint32_t>(c) << " changed under NFC");
    }
}
