// Bare-metal BPE tokenizer for Qwen3.
// Loads vocab and merges from an in-memory binary blob (tokenizer.bin).
// No file I/O, no libc stdio. Uses only malloc/memcpy/strcmp.

#ifndef TOKENIZER_BARE_H
#define TOKENIZER_BARE_H

#include <stdint.h>
#include <stddef.h>

/* Optional hook used by long-running bare-metal integrations.  Keep this
 * header usable by the standalone tokenizer tests, where UART is absent. */
#ifndef TOKENIZER_PROGRESS
#define TOKENIZER_PROGRESS(index, total, phase) ((void)0)
#endif

// Provided by bare_syscalls_uart.c
void *memcpy(void *dest, const void *src, size_t len);
void *memset(void *dest, int byte, size_t len);
int strcmp(const char *s1, const char *s2);
size_t strlen(const char *s);
// Provided by qwen3_bare_int8.c
void *malloc(size_t n);
void free(void *p);

#define TOK_HASH_SIZE 262144

typedef struct {
    const char *key;
    int id;
} TokVocabEntry;

typedef struct {
    const char *pair;
    int rank;
} TokMergeEntry;

typedef struct {
    // Vocab hash table (open addressing)
    TokVocabEntry *vocab_table;
    int vocab_table_cap;

    // Merge hash table (open addressing)
    TokMergeEntry *merge_table;
    int merge_table_cap;

    // Reverse lookup: id -> string
    const char **pieces_by_id;
    int vocab_size;

    // byte <-> unicode mapping for GPT BPE
    char byte_to_unicode[256][8];
    uint8_t unicode_to_byte_valid[65536];
    uint8_t unicode_to_byte[65536];
} BareTokenizer;

// ---- Hash function ----
static unsigned tok_hash(const char *s) {
    unsigned h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h;
}

// ---- UTF-8 helpers ----
static int tok_append_utf8(char *out, int cp) {
    if (cp < 0x80) { out[0] = (char)cp; out[1] = 0; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0|(cp>>6)); out[1] = (char)(0x80|(cp&0x3F)); out[2] = 0; return 2; }
    out[0] = (char)(0xE0|(cp>>12)); out[1] = (char)(0x80|((cp>>6)&0x3F)); out[2] = (char)(0x80|(cp&0x3F)); out[3] = 0;
    return 3;
}

static int tok_utf8_next(const char **s, int *cp) {
    const unsigned char *p = (const unsigned char *)(*s);
    if (!*p) return 0;
    if (p[0] < 0x80) { *cp = p[0]; *s += 1; return 1; }
    if ((p[0]&0xE0)==0xC0 && p[1]) { *cp = ((p[0]&0x1F)<<6)|(p[1]&0x3F); *s += 2; return 1; }
    if ((p[0]&0xF0)==0xE0 && p[1] && p[2]) { *cp = ((p[0]&0x0F)<<12)|((p[1]&0x3F)<<6)|(p[2]&0x3F); *s += 3; return 1; }
    *cp = p[0]; *s += 1; return 1;
}

// ---- byte <-> unicode init (GPT BPE standard) ----
static void tok_init_byte_unicode(BareTokenizer *t) {
    int bs[256], cs[256], nbs = 0;
    for (int i = 33; i <= 126; i++) bs[nbs++] = i;
    for (int i = 161; i <= 172; i++) bs[nbs++] = i;
    for (int i = 174; i <= 255; i++) bs[nbs++] = i;
    int ncs = nbs;
    for (int i = 0; i < nbs; i++) cs[i] = bs[i];
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        int found = 0;
        for (int i = 0; i < nbs; i++) { if (bs[i] == b) { found = 1; break; } }
        if (!found) { bs[nbs++] = b; cs[ncs++] = 256 + extra; extra++; }
    }
    memset(t->unicode_to_byte_valid, 0, sizeof(t->unicode_to_byte_valid));
    for (int i = 0; i < nbs; i++) {
        tok_append_utf8(t->byte_to_unicode[bs[i]], cs[i]);
        if (cs[i] >= 0 && cs[i] < 65536) {
            t->unicode_to_byte_valid[cs[i]] = 1;
            t->unicode_to_byte[cs[i]] = (uint8_t)bs[i];
        }
    }
}

// ---- Hash table operations (open addressing) ----
static void tok_vocab_insert(BareTokenizer *t, const char *key, int id) {
    unsigned h = tok_hash(key) % (unsigned)t->vocab_table_cap;
    while (t->vocab_table[h].key) {
        h = (h + 1) % (unsigned)t->vocab_table_cap;
    }
    t->vocab_table[h].key = key;
    t->vocab_table[h].id = id;
}

static int tok_vocab_lookup(BareTokenizer *t, const char *key) {
    unsigned h = tok_hash(key) % (unsigned)t->vocab_table_cap;
    while (t->vocab_table[h].key) {
        if (strcmp(t->vocab_table[h].key, key) == 0) return t->vocab_table[h].id;
        h = (h + 1) % (unsigned)t->vocab_table_cap;
    }
    return -1;
}

static void tok_merge_insert(BareTokenizer *t, const char *pair, int rank) {
    unsigned h = tok_hash(pair) % (unsigned)t->merge_table_cap;
    while (t->merge_table[h].pair) {
        h = (h + 1) % (unsigned)t->merge_table_cap;
    }
    t->merge_table[h].pair = pair;
    t->merge_table[h].rank = rank;
}

static int tok_merge_lookup(BareTokenizer *t, const char *pair) {
    unsigned h = tok_hash(pair) % (unsigned)t->merge_table_cap;
    while (t->merge_table[h].pair) {
        if (strcmp(t->merge_table[h].pair, pair) == 0) return t->merge_table[h].rank;
        h = (h + 1) % (unsigned)t->merge_table_cap;
    }
    return -1;
}

// ---- Safe unaligned read (RISC-V traps on misaligned loads) ----
static uint32_t read_u32(const char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

// ---- Load tokenizer from binary blob ----
// Binary format: see convert_tokenizer_bin.py
static __attribute__((unused)) BareTokenizer *tokenizer_load_from_mem(const char *blob) {
    if (blob[0] != 'T' || blob[1] != 'O' || blob[2] != 'K' || blob[3] != 'B') return NULL;

    const char *p = blob + 4;
    uint32_t vocab_size    = read_u32(p); p += 4;
    uint32_t merges_count  = read_u32(p); p += 4;
    uint32_t vocab_str_sz  = read_u32(p); p += 4;
    uint32_t merges_str_sz __attribute__((unused)) = read_u32(p); p += 4;

    BareTokenizer *t = (BareTokenizer *)malloc(sizeof(BareTokenizer));
    memset(t, 0, sizeof(BareTokenizer));

    tok_init_byte_unicode(t);

    // Allocate hash tables (2x size for open addressing)
    t->vocab_table_cap = (int)(vocab_size * 2);
    if (t->vocab_table_cap < TOK_HASH_SIZE) t->vocab_table_cap = TOK_HASH_SIZE;
    t->vocab_table = (TokVocabEntry *)malloc((size_t)t->vocab_table_cap * sizeof(TokVocabEntry));
    memset(t->vocab_table, 0, (size_t)t->vocab_table_cap * sizeof(TokVocabEntry));

    t->merge_table_cap = (int)(merges_count * 2);
    if (t->merge_table_cap < TOK_HASH_SIZE) t->merge_table_cap = TOK_HASH_SIZE;
    t->merge_table = (TokMergeEntry *)malloc((size_t)t->merge_table_cap * sizeof(TokMergeEntry));
    memset(t->merge_table, 0, (size_t)t->merge_table_cap * sizeof(TokMergeEntry));

    // Parse vocab entries: [id, offset, len] * vocab_size
    const char *vocab_entries = p;
    p += (size_t)vocab_size * 12;  // 3 * uint32 per entry

    // Vocab strings blob (zero-copy: point into blob)
    const char *vocab_strings = p;
    p += vocab_str_sz;

    // Reverse lookup table
    t->vocab_size = (int)vocab_size;
    t->pieces_by_id = (const char **)malloc((size_t)vocab_size * sizeof(const char *));
    memset(t->pieces_by_id, 0, (size_t)vocab_size * sizeof(const char *));

    for (uint32_t i = 0; i < vocab_size; i++) {
        uint32_t token_id  = read_u32(vocab_entries + i * 12);
        uint32_t offset    = read_u32(vocab_entries + i * 12 + 4);
        const char *key = vocab_strings + offset;
        tok_vocab_insert(t, key, (int)token_id);
        if ((int)token_id < t->vocab_size)
            t->pieces_by_id[token_id] = key;
        if ((i & 0x7fff) == 0)
            TOKENIZER_PROGRESS(i, vocab_size, "vocab");
    }

    // Parse merge entries: [offset, len] * merges_count
    const char *merge_entries = p;
    p += (size_t)merges_count * 8;  // 2 * uint32 per entry

    // Merges strings blob (zero-copy)
    const char *merges_strings = p;

    for (uint32_t i = 0; i < merges_count; i++) {
        uint32_t offset = read_u32(merge_entries + i * 8);
        const char *pair = merges_strings + offset;
        tok_merge_insert(t, pair, (int)i);
        if ((i & 0x7fff) == 0)
            TOKENIZER_PROGRESS(i, merges_count, "merges");
    }

    return t;
}

// ---- BPE encoding ----

// Convert raw bytes to BPE unicode representation
static char *tok_bytes_to_unicode(BareTokenizer *t, const char *text) {
    size_t n = strlen(text);
    size_t cap = n * 4 + 16, len = 0;
    char *out = (char *)malloc(cap);
    for (size_t i = 0; i < n; i++) {
        const char *rep = t->byte_to_unicode[(unsigned char)text[i]];
        size_t m = strlen(rep);
        if (len + m + 1 > cap) { cap = (cap + m + 16) * 2; char *no = (char *)malloc(cap); memcpy(no, out, len); free(out); out = no; }
        memcpy(out + len, rep, m);
        len += m;
    }
    out[len] = 0;
    return out;
}

// Simple dynamic string array
typedef struct { char **data; int size, cap; } TokStrVec;

static void tsv_init(TokStrVec *v) { v->data = NULL; v->size = v->cap = 0; }
static void tsv_push(TokStrVec *v, char *s) {
    if (v->size == v->cap) {
        int nc = v->cap ? v->cap * 2 : 8;
        char **nd = (char **)malloc((size_t)nc * sizeof(char *));
        if (v->data) { memcpy(nd, v->data, (size_t)v->size * sizeof(char *)); free(v->data); }
        v->data = nd; v->cap = nc;
    }
    v->data[v->size++] = s;
}
static void tsv_free(TokStrVec *v) {
    for (int i = 0; i < v->size; i++) free(v->data[i]);
    free(v->data);
}

__attribute__((unused))
static char *tok_strdup(const char *s) {
    size_t n = strlen(s);
    char *p = (char *)malloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *tok_concat2(const char *a, const char *b) {
    size_t na = strlen(a), nb = strlen(b);
    char *out = (char *)malloc(na + nb + 1);
    memcpy(out, a, na); memcpy(out + na, b, nb); out[na+nb] = 0;
    return out;
}

static char *tok_join_pair(const char *a, const char *b) {
    size_t na = strlen(a), nb = strlen(b);
    char *out = (char *)malloc(na + 1 + nb + 1);
    memcpy(out, a, na); out[na] = ' '; memcpy(out + na + 1, b, nb); out[na+1+nb] = 0;
    return out;
}

// BPE encode a single word
static int tok_bpe_word(BareTokenizer *t, const char *word, int *out_ids, int max_ids) {
    char *mapped = tok_bytes_to_unicode(t, word);

    // Split into UTF-8 characters
    TokStrVec parts;
    tsv_init(&parts);
    const char *pp = mapped;
    while (*pp) {
        const char *start = pp;
        int cp;
        if (!tok_utf8_next(&pp, &cp)) break;
        size_t n = (size_t)(pp - start);
        char *piece = (char *)malloc(n + 1);
        memcpy(piece, start, n); piece[n] = 0;
        tsv_push(&parts, piece);
    }
    free(mapped);

    // BPE merge loop
    while (parts.size >= 2) {
        int best_rank = 1 << 30, best_i = -1;
        for (int i = 0; i < parts.size - 1; i++) {
            char *pair = tok_join_pair(parts.data[i], parts.data[i+1]);
            int rank = tok_merge_lookup(t, pair);
            free(pair);
            if (rank >= 0 && rank < best_rank) { best_rank = rank; best_i = i; }
        }
        if (best_i < 0) break;
        char *merged = tok_concat2(parts.data[best_i], parts.data[best_i+1]);
        free(parts.data[best_i]);
        free(parts.data[best_i+1]);
        parts.data[best_i] = merged;
        for (int j = best_i+1; j < parts.size-1; j++) parts.data[j] = parts.data[j+1];
        parts.size--;
    }

    int count = 0;
    for (int i = 0; i < parts.size && count < max_ids; i++) {
        int id = tok_vocab_lookup(t, parts.data[i]);
        if (id >= 0) out_ids[count++] = id;
    }
    tsv_free(&parts);
    return count;
}

// print_uart is in uart.c (separate TU), print_uart_int_dec is static in
// uart_helper.c which is #included before this header in qwen3_bare_int8.c

// Encode text into token IDs
static int tokenizer_encode_bare(BareTokenizer *t, const char *text, int *out_ids, int max_ids) {
    int total = 0;
    const char *p = text;
    while (*p && total < max_ids) {
        const char *start = p;
        // Split on spaces: each word piece
        if (*p == ' ') {
            p++;
            while (*p == ' ') p++;
            while (*p && *p != ' ') p++;
        } else {
            while (*p && *p != ' ') p++;
        }
        size_t n = (size_t)(p - start);
        char *word = (char *)malloc(n + 1);
        memcpy(word, start, n); word[n] = 0;
        if (word[0]) {
            int cnt = tok_bpe_word(t, word, out_ids + total, max_ids - total);
            total += cnt;
        }
        free(word);
    }
    return total;
}

// Decode token IDs to text (output via UART)
static void tokenizer_decode_uart(BareTokenizer *t, int *tokens, int n) {
    extern void write_serial(char a);
    for (int i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= t->vocab_size || !t->pieces_by_id[tokens[i]])
            continue;
        const char *s = t->pieces_by_id[tokens[i]];
        const char *pp = s;
        while (*pp) {
            int cp;
            const char *old = pp;
            if (!tok_utf8_next(&pp, &cp)) break;
            if (cp >= 0 && cp < 65536 && t->unicode_to_byte_valid[cp]) {
                write_serial((char)t->unicode_to_byte[cp]);
            } else {
                while (old < pp) { write_serial(*old); old++; }
            }
        }
    }
}

#endif // TOKENIZER_BARE_H
