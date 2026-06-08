//! EVM memory expansion cost oracle.
//!
//! Computes the memory expansion gas cost per the Yellow Paper:
//! `G_memory(a) = 3*a + floor(a^2 / 512)` where `a` is the number
//! of 32-byte words. The expansion cost for going from old_size to
//! new_size is `G_memory(new_words) - G_memory(old_words)`.

pub fn memory_word_count(byte_size: u64) -> u64 {
    (byte_size + 31) / 32
}

pub fn memory_gas_cost(word_count: u64) -> u64 {
    3 * word_count + word_count * word_count / 512
}

pub fn memory_expansion_cost(old_byte_size: u64, new_byte_size: u64) -> u64 {
    if new_byte_size <= old_byte_size {
        return 0;
    }
    let old_words = memory_word_count(old_byte_size);
    let new_words = memory_word_count(new_byte_size);
    memory_gas_cost(new_words) - memory_gas_cost(old_words)
}

pub fn required_memory_size(offset: u64, size: u64) -> u64 {
    if size == 0 { return 0; }
    offset.saturating_add(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_count() {
        assert_eq!(memory_word_count(0), 0);
        assert_eq!(memory_word_count(1), 1);
        assert_eq!(memory_word_count(32), 1);
        assert_eq!(memory_word_count(33), 2);
        assert_eq!(memory_word_count(64), 2);
    }

    #[test]
    fn gas_cost_small() {
        assert_eq!(memory_gas_cost(0), 0);
        assert_eq!(memory_gas_cost(1), 3); // 3*1 + 1/512 = 3
        assert_eq!(memory_gas_cost(2), 6); // 3*2 + 4/512 = 6
        assert_eq!(memory_gas_cost(32), 98); // 3*32 + 1024/512 = 96+2 = 98
    }

    #[test]
    fn expansion_cost_zero_to_32_bytes() {
        let cost = memory_expansion_cost(0, 32);
        assert_eq!(cost, 3); // 1 word, cost = 3
    }

    #[test]
    fn expansion_cost_no_expansion() {
        assert_eq!(memory_expansion_cost(32, 32), 0);
        assert_eq!(memory_expansion_cost(64, 32), 0);
    }

    #[test]
    fn expansion_cost_32_to_64() {
        // 1 word → 2 words: cost(2) - cost(1) = 6 - 3 = 3
        assert_eq!(memory_expansion_cost(32, 64), 3);
    }

    #[test]
    fn expansion_cost_large() {
        // 0 → 1024 bytes = 32 words
        // cost(32) = 3*32 + 32*32/512 = 96 + 2 = 98
        assert_eq!(memory_expansion_cost(0, 1024), 98);
    }

    #[test]
    fn required_memory_size_zero_size() {
        assert_eq!(required_memory_size(100, 0), 0);
    }

    #[test]
    fn required_memory_size_normal() {
        assert_eq!(required_memory_size(0, 32), 32);
        assert_eq!(required_memory_size(10, 22), 32);
    }
}
