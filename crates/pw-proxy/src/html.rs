//! Pure transforms for proxied responses: decoding, script injection, CSP rewriting,
//! and channel names. Kept apart from the server so each rule has a unit test.

use std::io::Read;

pub const RUNTIME_PATH: &str = "/__pw/runtime.js";
pub const RUNTIME_JS: &str = include_str!("../../../runtime/runtime.js");
pub const EDITOR_JS: &str = include_str!("../../../runtime/editor.js");
pub const DEFAULT_CHANNEL: &str = "production";

pub fn find_ascii_ci(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w.eq_ignore_ascii_case(needle))
}

/// Nonces come from a response header, so only base64-like characters go into the HTML.
fn safe_nonce(n: &str) -> Option<&str> {
    let ok = !n.is_empty() && n.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '-' | '_'));
    ok.then_some(n)
}

pub fn script_tag(nonce: Option<&str>) -> String {
    match nonce.and_then(safe_nonce) {
        Some(n) => format!("<script src=\"{RUNTIME_PATH}\" nonce=\"{n}\" defer></script>"),
        None => format!("<script src=\"{RUNTIME_PATH}\" defer></script>"),
    }
}

/// Insert the runtime tag before `</head>`, else after `<body ...>`, else at the start.
/// Idempotent: a page that already loads the runtime is returned unchanged.
pub fn inject_runtime(html: &[u8], nonce: Option<&str>) -> Vec<u8> {
    if find_ascii_ci(html, RUNTIME_PATH.as_bytes()).is_some() {
        return html.to_vec();
    }
    let tag = script_tag(nonce);
    let at = match find_ascii_ci(html, b"</head>") {
        Some(p) => p,
        None => match find_ascii_ci(html, b"<body") {
            Some(p) => html[p..].iter().position(|&c| c == b'>').map_or(0, |q| p + q + 1),
            None => 0,
        },
    };
    let mut out = Vec::with_capacity(html.len() + tag.len());
    out.extend_from_slice(&html[..at]);
    out.extend_from_slice(tag.as_bytes());
    out.extend_from_slice(&html[at..]);
    out
}

/// Decode a response body. `Ok(None)` means an encoding we do not decode; pass it through untouched.
pub fn decode(body: &[u8], encoding: Option<&str>) -> std::io::Result<Option<Vec<u8>>> {
    let enc = encoding.unwrap_or("").trim().to_ascii_lowercase();
    let mut out = Vec::new();
    match enc.as_str() {
        "" | "identity" => return Ok(Some(body.to_vec())),
        "gzip" | "x-gzip" => {
            flate2::read::GzDecoder::new(body).read_to_end(&mut out)?;
        }
        "deflate" => {
            if flate2::read::ZlibDecoder::new(body).read_to_end(&mut out).is_err() {
                out.clear();
                flate2::read::DeflateDecoder::new(body).read_to_end(&mut out)?;
            }
        }
        "br" => {
            brotli::Decompressor::new(body, 4096).read_to_end(&mut out)?;
        }
        _ => return Ok(None),
    }
    Ok(Some(out))
}

pub struct CspOutcome {
    pub header: String,
    /// A nonce the runtime tag can reuse, when the policy already names one.
    pub nonce: Option<String>,
}

fn tokens_after_name(directive: &str) -> Vec<String> {
    directive.split_whitespace().skip(1).map(String::from).collect()
}

fn name_of(directive: &str) -> &str {
    directive.split_whitespace().next().unwrap_or("")
}

/// Let the runtime load and beacon under this policy, and change nothing else.
/// - script-src (or default-src when script-src is absent) gets `'self'` unless a nonce
///   is present, in which case the runtime tag carries that nonce.
/// - connect-src (or default-src) gets `'self'` so beacons can be sent.
pub fn adjust_csp(value: &str) -> CspOutcome {
    let mut directives: Vec<String> = value.split(';').map(|d| d.trim().to_string()).filter(|d| !d.is_empty()).collect();
    let index_of = |name: &str, dirs: &[String]| dirs.iter().position(|d| name_of(d) == name);
    let script_idx = index_of("script-src", &directives);
    let default_idx = index_of("default-src", &directives);
    let mut nonce = None;

    if let Some(i) = script_idx.or(default_idx) {
        let tokens = tokens_after_name(&directives[i]);
        nonce = tokens
            .iter()
            .find_map(|t| t.strip_prefix("'nonce-").and_then(|r| r.strip_suffix('\'')).map(String::from));
        if nonce.is_none() && !tokens.iter().any(|t| t == "'self'") {
            if script_idx.is_some() {
                directives[i].push_str(" 'self'");
            } else {
                let mut widened: Vec<String> = tokens.into_iter().filter(|t| t != "'none'").collect();
                widened.push("'self'".into());
                directives.push(format!("script-src {}", widened.join(" ")));
            }
        }
    }

    match index_of("connect-src", &directives) {
        Some(i) => {
            if !tokens_after_name(&directives[i]).iter().any(|t| t == "'self'") {
                directives[i].push_str(" 'self'");
            }
        }
        None => {
            if let Some(d) = default_idx {
                let mut tokens: Vec<String> = tokens_after_name(&directives[d]).into_iter().filter(|t| t != "'none'").collect();
                if !tokens.iter().any(|t| t == "'self'") {
                    tokens.push("'self'".into());
                }
                directives.push(format!("connect-src {}", tokens.join(" ")));
            }
        }
    }
    CspOutcome { header: directives.join("; "), nonce }
}

/// Channels are preview/feature names: segments of [A-Za-z0-9._-], no empty or dot-only segments.
pub fn valid_channel(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.split('/').all(|seg| {
            !seg.is_empty() && seg != "." && seg != ".." && seg.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
}

pub fn channel_from_cookie(header: Option<&str>) -> String {
    header
        .and_then(|h| h.split(';').map(str::trim).find_map(|kv| kv.strip_prefix("pw_channel=")))
        .filter(|v| valid_channel(v))
        .map(String::from)
        .unwrap_or_else(|| DEFAULT_CHANNEL.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    const PAGE: &[u8] = b"<!doctype html><html><head><title>Shop</title></head><body><button>Buy</button></body></html>";

    #[test]
    fn injects_before_head_close() {
        let out = String::from_utf8(inject_runtime(PAGE, None)).unwrap();
        assert!(out.contains("<script src=\"/__pw/runtime.js\" defer></script></head>"), "{out}");
    }

    #[test]
    fn injection_is_idempotent() {
        let once = inject_runtime(PAGE, None);
        assert_eq!(inject_runtime(&once, None), once);
    }

    #[test]
    fn falls_back_to_after_body_open_then_to_the_start() {
        let no_head = b"<html><body class=x><p>hi</p></body></html>";
        let out = String::from_utf8(inject_runtime(no_head, None)).unwrap();
        assert!(out.starts_with("<html><body class=x><script"), "{out}");
        let bare = b"just text";
        assert!(String::from_utf8(inject_runtime(bare, None)).unwrap().starts_with("<script"));
    }

    #[test]
    fn reuses_a_policy_nonce_on_the_runtime_tag() {
        let out = String::from_utf8(inject_runtime(PAGE, Some("abc123=="))).unwrap();
        assert!(out.contains("nonce=\"abc123==\""));
    }

    #[test]
    fn refuses_nonces_that_could_break_out_of_the_attribute() {
        let tag = script_tag(Some("x\" onload=\"alert(1)"));
        assert!(!tag.contains("onload"), "{tag}");
    }

    #[test]
    fn decodes_gzip_brotli_and_passes_unknown_encodings_through() {
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(PAGE).unwrap();
        let gz = enc.finish().unwrap();
        assert_eq!(decode(&gz, Some("gzip")).unwrap().unwrap(), PAGE);

        let mut br = Vec::new();
        {
            let mut w = brotli::CompressorWriter::new(&mut br, 4096, 5, 22);
            w.write_all(PAGE).unwrap();
        }
        assert_eq!(decode(&br, Some("br")).unwrap().unwrap(), PAGE);

        assert_eq!(decode(PAGE, Some("zstd")).unwrap(), None);
        assert_eq!(decode(PAGE, None).unwrap().unwrap(), PAGE);
    }

    #[test]
    fn corrupt_gzip_is_an_error_not_a_silent_page() {
        assert!(decode(b"not gzip at all", Some("gzip")).is_err());
    }

    #[test]
    fn csp_without_script_rules_is_left_alone_apart_from_beacons() {
        let out = adjust_csp("img-src 'self'");
        assert_eq!(out.header, "img-src 'self'");
        assert!(out.nonce.is_none());
    }

    #[test]
    fn csp_script_src_gets_self_when_there_is_no_nonce() {
        let out = adjust_csp("script-src https://cdn.example");
        assert!(out.header.contains("script-src https://cdn.example 'self'"), "{}", out.header);
        // No default-src means connect-src is unrestricted already, so beacons need no change.
        assert!(!out.header.contains("connect-src"), "{}", out.header);
    }

    #[test]
    fn csp_with_a_nonce_reuses_it_and_does_not_widen() {
        let out = adjust_csp("script-src 'nonce-Zm9v' 'strict-dynamic'; default-src 'self'");
        assert_eq!(out.nonce.as_deref(), Some("Zm9v"));
        assert!(!out.header.contains("script-src 'nonce-Zm9v' 'strict-dynamic' 'self'"));
    }

    #[test]
    fn csp_default_src_only_gets_a_script_src_copy_with_self() {
        let out = adjust_csp("default-src 'none'");
        assert!(out.header.contains("script-src 'self'"), "{}", out.header);
        assert!(out.header.contains("connect-src 'self'"), "{}", out.header);
        assert!(!out.header.contains("'none' 'self'"), "'none' must not be combined with 'self'");
    }

    #[test]
    fn channel_names_are_checked() {
        assert!(valid_channel("production"));
        assert!(valid_channel("preview/feature-x"));
        assert!(!valid_channel("../etc"));
        assert!(!valid_channel("preview//x"));
        assert!(!valid_channel("a b"));
        assert!(!valid_channel(""));
    }

    #[test]
    fn channel_comes_from_the_cookie_or_defaults() {
        assert_eq!(channel_from_cookie(Some("a=1; pw_channel=preview/x")), "preview/x");
        assert_eq!(channel_from_cookie(Some("pw_channel=../x")), "production");
        assert_eq!(channel_from_cookie(None), "production");
    }
}
