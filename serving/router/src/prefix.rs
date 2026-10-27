//! Deciding what "the same prefix" means for an OpenAI chat request.
//!
//! The engines cache KV for the longest shared token prefix, so requests that begin with the same system prompt (and few-shot turns)
//! should land on the same replica. The router sees only JSON, not tokens, so it keys on message content:
//!   1. an explicit `X-Prefix-Key` header, if the caller knows better;
//!   2. else the hash of the first `prefix_messages` messages, if configured;
//!   3. else the hash of all `system` messages (what agents typically share);
//!   4. else the first message, else the model name.

use crate::ring::hash64;
use axum::http::HeaderMap;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    Header,
    FirstMessages,
    System,
    FirstMessage,
    Model,
    None,
}

pub fn prefix_key(headers: &HeaderMap, body: Option<&Value>, prefix_messages: Option<usize>) -> (u64, KeySource) {
    if let Some(v) = headers.get("x-prefix-key").and_then(|v| v.to_str().ok()) {
        return (hash64(v.as_bytes()), KeySource::Header);
    }
    let Some(body) = body else { return (0, KeySource::None) };
    let messages = body.get("messages").and_then(|m| m.as_array());
    if let Some(msgs) = messages {
        if let Some(n) = prefix_messages {
            return (hash_messages(msgs.iter().take(n)), KeySource::FirstMessages);
        }
        let system: Vec<&Value> = msgs.iter().filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system")).collect();
        if !system.is_empty() {
            return (hash_messages(system.into_iter()), KeySource::System);
        }
        if let Some(first) = msgs.first() {
            return (hash_messages(std::iter::once(first)), KeySource::FirstMessage);
        }
    }
    match body.get("model").and_then(|m| m.as_str()) {
        Some(m) => (hash64(m.as_bytes()), KeySource::Model),
        None => (0, KeySource::None),
    }
}

/// Hash message objects by their serialised form. Clients serialise deterministically, so equal prompts give equal bytes;
/// image parts are hashed through their URL/data-URI string like any other content.
fn hash_messages<'a>(msgs: impl Iterator<Item = &'a Value>) -> u64 {
    let mut buf = Vec::new();
    for m in msgs {
        buf.extend_from_slice(m.to_string().as_bytes());
        buf.push(0x1e);
    }
    hash64(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req(system: &str, user: &str) -> Value {
        json!({"model": "m", "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                                           {"type": "text", "text": user}]}]})
    }

    #[test]
    fn same_system_prompt_gives_the_same_key_whatever_the_user_turn() {
        let h = HeaderMap::new();
        let (a, sa) = prefix_key(&h, Some(&req("be an agent", "click Save")), None);
        let (b, _) = prefix_key(&h, Some(&req("be an agent", "click Cancel")), None);
        let (c, _) = prefix_key(&h, Some(&req("be a different agent", "click Save")), None);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(sa, KeySource::System);
    }

    #[test]
    fn explicit_header_wins() {
        let mut h = HeaderMap::new();
        h.insert("x-prefix-key", "session-42".parse().unwrap());
        let (k, src) = prefix_key(&h, Some(&req("s", "u")), None);
        assert_eq!(src, KeySource::Header);
        assert_eq!(k, hash64(b"session-42"));
    }

    #[test]
    fn first_n_messages_includes_few_shot_turns() {
        let h = HeaderMap::new();
        let mk = |shot: &str| json!({"messages": [
            {"role": "system", "content": "s"}, {"role": "user", "content": shot},
            {"role": "assistant", "content": "(1, 2)"}, {"role": "user", "content": "current"}]});
        let (a, s1) = prefix_key(&h, Some(&mk("example A")), Some(3));
        let (b, _) = prefix_key(&h, Some(&mk("example B")), Some(3));
        assert_eq!(s1, KeySource::FirstMessages);
        assert_ne!(a, b, "different few-shot examples are different prefixes when they are part of the key");
        let (c, _) = prefix_key(&h, Some(&mk("example A")), Some(1));
        let (d, _) = prefix_key(&h, Some(&mk("example B")), Some(1));
        assert_eq!(c, d, "with only the system message hashed the few-shot turn does not matter");
    }

    #[test]
    fn falls_back_sensibly() {
        let h = HeaderMap::new();
        let no_system = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(prefix_key(&h, Some(&no_system), None).1, KeySource::FirstMessage);
        let only_model = json!({"model": "qwen"});
        assert_eq!(prefix_key(&h, Some(&only_model), None).1, KeySource::Model);
        assert_eq!(prefix_key(&h, None, None), (0, KeySource::None));
        assert_eq!(prefix_key(&h, Some(&json!({})), None).1, KeySource::None);
    }
}
