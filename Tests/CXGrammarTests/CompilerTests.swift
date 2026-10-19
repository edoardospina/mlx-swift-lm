// Copyright © 2026 Apple Inc.
//
// Direct C-API tests for the CXGrammar shim's tokenizer-aware
// GrammarCompiler path. Distinct from SchemaErrorTests, which covers
// the tokenizer-free `xg_grammar_from_json_schema` entry point —
// GrammarCompiler is the path that binds a schema to a specific
// tokenizer so a matcher can be built from it.

import Foundation
import MLXCXGrammar
import Testing

@Suite
struct CompilerTests {

    /// Compile a minimal JSON schema against the synthetic tokenizer.
    ///
    /// Builds the same synthetic gemma-3 vocab used in
    /// TokenizerInfoTests, constructs an `XGGrammarCompiler`, then
    /// compiles `{"type":"object","properties":{"name":{"type":"string"}}}`
    /// against it. The bar is: `XG_OK` + non-null `XGCompiledGrammar`
    /// handle.
    @Test
    func testCompileSimpleSchema() throws {
        let fixture = try Self.loadGemmaFixture()

        let vocabSize = Int(fixture.vocabSize)
        let eosId = Int(fixture.eosTokenId)

        let placeholder = "<|tok|>"
        var vocabStrings = Array(repeating: placeholder, count: vocabSize)
        if eosId >= 0 && eosId < vocabSize {
            vocabStrings[eosId] = fixture.eosTokenString
        }

        let cStrings = vocabStrings.map { $0.utf8CString }
        var vocabPtrs: [UnsafePointer<CChar>?] = cStrings.map { arr in
            arr.withUnsafeBufferPointer { buf in buf.baseAddress }
        }

        var info: OpaquePointer?
        let stopTokens: [Int32] = [Int32(eosId)]

        let tokenizerStatus: XGStatus = vocabPtrs.withUnsafeMutableBufferPointer { vocabBuf in
            stopTokens.withUnsafeBufferPointer { stopBuf in
                xg_tokenizer_info_new(
                    vocabBuf.baseAddress,
                    vocabBuf.count,
                    XG_VOCAB_TYPE_RAW,
                    stopBuf.baseAddress,
                    stopBuf.count,
                    &info
                )
            }
        }
        #expect(tokenizerStatus == XG_OK, "precondition: xg_tokenizer_info_new should succeed")
        #expect(info != nil, "precondition: xg_tokenizer_info_new should produce a handle")
        defer { xg_tokenizer_info_free(info) }

        var compiler: OpaquePointer?
        let compilerStatus = xg_grammar_compiler_new(info, &compiler)
        #expect(
            compilerStatus == XG_OK,
            "xg_grammar_compiler_new returned \(compilerStatus); last error: \(xg_last_error_message().map { String(cString: $0) } ?? "<nil>")"
        )
        #expect(compiler != nil, "xg_grammar_compiler_new produced a null handle on success")
        defer { xg_grammar_compiler_free(compiler) }

        let schema = #"{"type":"object","properties":{"name":{"type":"string"}}}"#

        var compiled: OpaquePointer?
        let compileStatus: XGStatus = schema.withCString { schemaPtr in
            xg_compile_json_schema(compiler, schemaPtr, &compiled)
        }

        #expect(
            compileStatus == XG_OK,
            "xg_compile_json_schema returned \(compileStatus); last error: \(xg_last_error_message().map { String(cString: $0) } ?? "<nil>")"
        )
        #expect(compiled != nil, "xg_compile_json_schema produced a null handle on success")
        xg_compiled_grammar_free(compiled)

        _ = cStrings.count
    }

    /// `max_whitespace_cnt` bounds each gap between two tokens.
    ///
    /// Over a vocabulary of one-character tokens, the schema
    /// `{"a": [string], "b": [string]}` with both required, and the
    /// matcher past `{"a":[]`: capped at 3, a space is allowed three
    /// times and then only `,` — never `}`, since `b` is still owed —
    /// and past the `,` three times more and then only `"`. A model whose
    /// favourite is a token the grammar refuses at that point can
    /// otherwise write whitespace until its token limit.
    @Test
    func testWhitespaceCapBoundsEachGap() throws {
        let matcher = try SmallVocabMatcher(maxWhitespaceCount: 3)
        for piece in ["{", "\"", "a", "\"", ":", "[", "]"] { try matcher.accept(piece) }

        for _ in 0 ..< 3 {
            #expect(try matcher.allows(" "))
            #expect(!(try matcher.allows("}")))
            try matcher.accept(" ")
        }
        #expect(!(try matcher.allows(" ")))
        #expect(!(try matcher.allows("\n")))
        #expect(try matcher.allows(","))
        #expect(!(try matcher.allows("}")))

        try matcher.accept(",")
        for _ in 0 ..< 3 { try matcher.accept("\n") }
        #expect(!(try matcher.allows(" ")))
        #expect(try matcher.allows("\""))
    }

    /// With no cap, and with a count of 0 or less, a gap holds any
    /// amount of whitespace, as `xg_compile_json_schema`'s does.
    @Test(arguments: [nil, 0, -1] as [Int32?])
    func testWhitespaceUnboundedWithoutACap(maxWhitespaceCount: Int32?) throws {
        let matcher = try SmallVocabMatcher(maxWhitespaceCount: maxWhitespaceCount)
        for piece in ["{", "\"", "a", "\"", ":", "[", "]"] { try matcher.accept(piece) }
        for _ in 0 ..< 12 { try matcher.accept(" ") }
        #expect(try matcher.allows(" "))
        #expect(try matcher.allows(","))
    }

    /// One separator without the other is refused, never half-applied.
    @Test
    func testOneSeparatorAloneIsRefused() throws {
        let matcher = try SmallVocabMatcher(maxWhitespaceCount: nil)
        var compiled: OpaquePointer?
        let status: XGStatus = Self.capSchema.withCString { schemaPtr in
            ",".withCString { itemPtr in
                xg_compile_json_schema_ex(
                    matcher.compiler, schemaPtr, 0, -1, itemPtr, nil, 1, 0, &compiled)
            }
        }
        #expect(status == XG_ERR_INVALID_ARG)
        #expect(compiled == nil)
    }

    // MARK: - A matcher over one-character tokens

    fileprivate static let capSchema = #"""
        {"type":"object","properties":{"a":{"type":"array","items":{"type":"string"}},"b":{"type":"array","items":{"type":"string"}}},"required":["a","b"]}
        """#

    /// A matcher for `capSchema` over a raw vocabulary of single JSON
    /// characters, compiled by `xg_compile_json_schema_ex` with
    /// `any_whitespace` on — or by `xg_compile_json_schema` when the
    /// count is `nil` — that reads its mask piece by piece.
    fileprivate final class SmallVocabMatcher {
        static let vocab = ["{", "}", "[", "]", ",", ":", "\"", "a", "b", " ", "\n", "</s>"]

        let info: OpaquePointer?
        let compiler: OpaquePointer?
        let compiled: OpaquePointer?
        let matcher: OpaquePointer?

        init(maxWhitespaceCount: Int32?) throws {
            let cStrings = Self.vocab.map { Array($0.utf8CString) }
            var vocabPtrs: [UnsafePointer<CChar>?] = cStrings.map {
                $0.withUnsafeBufferPointer { $0.baseAddress }
            }
            let stopTokens: [Int32] = [Int32(Self.vocab.count - 1)]
            var info: OpaquePointer?
            let infoStatus: XGStatus = vocabPtrs.withUnsafeMutableBufferPointer { vocabBuf in
                stopTokens.withUnsafeBufferPointer { stopBuf in
                    xg_tokenizer_info_new(
                        vocabBuf.baseAddress, vocabBuf.count, XG_VOCAB_TYPE_RAW,
                        stopBuf.baseAddress, stopBuf.count, &info)
                }
            }
            _ = cStrings.count
            try Self.check(infoStatus, "xg_tokenizer_info_new")
            self.info = info

            var compiler: OpaquePointer?
            try Self.check(xg_grammar_compiler_new(info, &compiler), "xg_grammar_compiler_new")
            self.compiler = compiler

            var compiled: OpaquePointer?
            let compileStatus: XGStatus = CompilerTests.capSchema.withCString { schemaPtr in
                if let maxWhitespaceCount {
                    return xg_compile_json_schema_ex(
                        compiler, schemaPtr, 1, -1, nil, nil, 1, maxWhitespaceCount, &compiled)
                }
                return xg_compile_json_schema(compiler, schemaPtr, &compiled)
            }
            try Self.check(compileStatus, "schema compile")
            self.compiled = compiled

            var matcher: OpaquePointer?
            try Self.check(xg_matcher_new(compiled, &matcher), "xg_matcher_new")
            self.matcher = matcher
        }

        deinit {
            xg_matcher_free(matcher)
            xg_compiled_grammar_free(compiled)
            xg_grammar_compiler_free(compiler)
            xg_tokenizer_info_free(info)
        }

        /// Whether the next token may be `piece`.
        func allows(_ piece: String) throws -> Bool {
            let vocabSize = Int32(Self.vocab.count)
            var mask = [Int32](repeating: 0, count: Int(xg_bitmask_size(vocabSize)))
            let status = mask.withUnsafeMutableBufferPointer { buf in
                xg_matcher_fill_next_token_bitmask(
                    matcher, buf.baseAddress, buf.count, vocabSize, nil)
            }
            try Self.check(status, "xg_matcher_fill_next_token_bitmask")
            let id = try Self.id(of: piece)
            return (UInt32(bitPattern: mask[id / 32]) >> UInt32(id % 32)) & 1 == 1
        }

        func accept(_ piece: String) throws {
            try Self.check(
                xg_matcher_accept_token(matcher, Int32(try Self.id(of: piece))),
                "accept \(piece.debugDescription)")
        }

        private static func id(of piece: String) throws -> Int {
            guard let id = vocab.firstIndex(of: piece) else {
                throw FixtureError.malformed("no token \(piece.debugDescription)")
            }
            return id
        }

        private static func check(_ status: XGStatus, _ step: String) throws {
            guard status == XG_OK else {
                let message = xg_last_error_message().map { String(cString: $0) } ?? "<nil>"
                throw FixtureError.malformed("\(step) returned \(status): \(message)")
            }
        }
    }

    // MARK: - Fixture loading

    private struct GemmaFixture {
        let vocabSize: Int
        let eosTokenId: Int
        let eosTokenString: String
    }

    private static func loadGemmaFixture() throws -> GemmaFixture {
        let url = Self.goldensDirectory.appendingPathComponent("tokenizer_gemma3.json")
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let json else {
            throw FixtureError.malformed("top-level not an object")
        }
        guard let vocabSize = json["vocabSize"] as? Int else {
            throw FixtureError.malformed("missing vocabSize")
        }
        guard let eosTokenId = json["eosTokenId"] as? Int else {
            throw FixtureError.malformed("missing eosTokenId")
        }
        guard let eosTokenString = json["eosTokenString"] as? String else {
            throw FixtureError.malformed("missing eosTokenString")
        }
        return GemmaFixture(
            vocabSize: vocabSize,
            eosTokenId: eosTokenId,
            eosTokenString: eosTokenString
        )
    }

    private static let goldensDirectory: URL = {
        let thisFile = URL(fileURLWithPath: #filePath)
        return
            thisFile
            .deletingLastPathComponent()  // Tests/CXGrammarTests
            .appendingPathComponent("Fixtures", isDirectory: true)
            .appendingPathComponent("goldens", isDirectory: true)
    }()

    private enum FixtureError: Error {
        case malformed(String)
    }
}
