//! File digests, for records that must name the exact bytes they were made from.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::Result;

/// SHA-256 of a file's bytes, as lowercase hex. The file is read in chunks, so a large model is never held in memory.
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn the_digest_is_the_sha256_of_the_bytes() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("hash-{}.bin", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        // SHA-256("abc"), the standard test vector.
        assert_eq!(file_sha256(&path).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let _ = std::fs::remove_file(&path);
    }
}
