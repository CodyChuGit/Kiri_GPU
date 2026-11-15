//! Enough YAML to read CDI specs, as `nvidia-ctk cdi generate` and other
//! vendors' tools (all through Go's yaml packages) write them: block
//! mappings and sequences, plain and quoted scalars, flow sequences and
//! mappings of scalars, comments and a leading `---`. Anchors, tags,
//! multi-document streams and block scalars (`|`, `>`) are refused with an
//! error rather than misread. A JSON spec is read by serde_json instead.
//! Kept here rather than as a dependency: this is all a spec needs.

use serde_json::{Map, Value};

#[derive(Debug)]
struct Line<'a> {
    indent: usize,
    text: &'a str,
    no: usize,
}

/// Parses `text` as one YAML document.
pub fn parse(text: &str) -> Result<Value, String> {
    let mut lines = Vec::new();
    let mut started = false;
    for (i, raw) in text.lines().enumerate() {
        let body = strip_comment(raw).trim_end();
        let trimmed = body.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        if body[..body.len() - trimmed.len()].contains('\t') {
            return Err(format!("line {}: tabs in indentation", i + 1));
        }
        if trimmed == "---" && body.len() == 3 {
            if started {
                return Err(format!("line {}: more than one document", i + 1));
            }
            started = true;
            continue;
        }
        started = true;
        lines.push(Line { indent: body.len() - trimmed.len(), text: trimmed, no: i + 1 });
    }
    if lines.is_empty() {
        return Ok(Value::Null);
    }
    let mut p = Parser { lines, at: 0 };
    let indent = p.lines[0].indent;
    let v = p.node(indent)?;
    if let Some(l) = p.lines.get(p.at) {
        return Err(format!("line {}: unexpected indentation", l.no));
    }
    Ok(v)
}

/// The line without a trailing ` # comment` (outside quotes).
fn strip_comment(line: &str) -> &str {
    let (mut single, mut double) = (false, false);
    let b = line.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        match c {
            b'\'' if !double => single = !single,
            b'"' if !single && (i == 0 || b[i - 1] != b'\\') => double = !double,
            b'#' if !single && !double && (i == 0 || b[i - 1] == b' ' || b[i - 1] == b'\t') => return &line[..i],
            _ => {}
        }
    }
    line
}

struct Parser<'a> {
    lines: Vec<Line<'a>>,
    at: usize,
}

fn is_item(text: &str) -> bool {
    text == "-" || text.starts_with("- ")
}

impl<'a> Parser<'a> {
    fn node(&mut self, indent: usize) -> Result<Value, String> {
        if is_item(self.lines[self.at].text) { self.sequence(indent) } else { self.mapping(indent) }
    }

    fn sequence(&mut self, indent: usize) -> Result<Value, String> {
        let mut out = Vec::new();
        while let Some(l) = self.lines.get(self.at) {
            if l.indent != indent || !is_item(l.text) {
                if l.indent > indent {
                    return Err(format!("line {}: unexpected indentation", l.no));
                }
                break;
            }
            let rest = l.text[1..].trim_start();
            if rest.is_empty() {
                self.at += 1;
                match self.lines.get(self.at) {
                    Some(n) if n.indent > indent => {
                        let i = n.indent;
                        out.push(self.node(i)?);
                    }
                    _ => out.push(Value::Null),
                }
                continue;
            }
            if is_item(rest) || key_split(rest).is_some() {
                // `- key: value` (or `- - x`): a node starting on this line,
                // indented to where its text starts.
                let col = l.indent + (l.text.len() - rest.len());
                self.lines[self.at] = Line { indent: col, text: rest, no: l.no };
                out.push(self.node(col)?);
                continue;
            }
            let no = l.no;
            self.at += 1;
            out.push(scalar(rest).map_err(|e| format!("line {no}: {e}"))?);
        }
        Ok(Value::Array(out))
    }

    fn mapping(&mut self, indent: usize) -> Result<Value, String> {
        let mut out = Map::new();
        while let Some(l) = self.lines.get(self.at) {
            if l.indent != indent || is_item(l.text) {
                if l.indent > indent {
                    return Err(format!("line {}: unexpected indentation", l.no));
                }
                break;
            }
            let no = l.no;
            let (key, value) = key_split(l.text).ok_or_else(|| format!("line {no}: expected `key: value`"))?;
            let key = match scalar(key).map_err(|e| format!("line {no}: {e}"))? {
                Value::String(s) => s,
                other => other.to_string(),
            };
            self.at += 1;
            let v = if value.is_empty() {
                match self.lines.get(self.at) {
                    Some(n) if n.indent > indent => {
                        let i = n.indent;
                        self.node(i)?
                    }
                    // Go's yaml puts a key's sequence at the key's own indentation.
                    Some(n) if n.indent == indent && is_item(n.text) => self.sequence(indent)?,
                    _ => Value::Null,
                }
            } else {
                scalar(value).map_err(|e| format!("line {no}: {e}"))?
            };
            if out.insert(key.clone(), v).is_some() {
                return Err(format!("line {no}: key {key:?} given twice"));
            }
        }
        Ok(Value::Object(out))
    }
}

/// `key: value` or `key:` split at the first `:` (outside quotes) that
/// ends the line or is followed by a space.
fn key_split(text: &str) -> Option<(&str, &str)> {
    let b = text.as_bytes();
    let mut i = 0;
    if matches!(b.first(), Some(b'"' | b'\'')) {
        let q = b[0];
        i = 1;
        while i < b.len() && !(b[i] == q && (q == b'\'' || b[i - 1] != b'\\')) {
            i += 1;
        }
        i += 1;
    }
    if matches!(b.first(), Some(b'[' | b'{')) {
        return None;
    }
    while i < b.len() {
        if b[i] == b':' && (i + 1 == b.len() || b[i + 1] == b' ') {
            return Some((text[..i].trim_end(), text[i + 1..].trim()));
        }
        i += 1;
    }
    None
}

/// One scalar, or a flow sequence or mapping of scalars.
fn scalar(s: &str) -> Result<Value, String> {
    let s = s.trim();
    match s.as_bytes().first() {
        Some(b'"') => return double_quoted(s).map(Value::String),
        Some(b'\'') => {
            let inner = s.strip_prefix('\'').and_then(|x| x.strip_suffix('\'')).filter(|_| s.len() >= 2).ok_or("unterminated quote")?;
            return Ok(Value::String(inner.replace("''", "'")));
        }
        Some(b'[') => {
            let inner = s.strip_suffix(']').ok_or("unterminated [")?[1..].trim();
            if inner.is_empty() {
                return Ok(Value::Array(Vec::new()));
            }
            return split_flow(inner)?.into_iter().map(scalar).collect::<Result<Vec<_>, _>>().map(Value::Array);
        }
        Some(b'{') => {
            let inner = s.strip_suffix('}').ok_or("unterminated {")?[1..].trim();
            let mut m = Map::new();
            if !inner.is_empty() {
                for part in split_flow(inner)? {
                    let (k, v) = key_split(part).ok_or_else(|| format!("bad flow entry {part:?}"))?;
                    let k = match scalar(k)? {
                        Value::String(s) => s,
                        other => other.to_string(),
                    };
                    m.insert(k, scalar(v)?);
                }
            }
            return Ok(Value::Object(m));
        }
        Some(b'|' | b'>') => return Err("block scalars are not supported".into()),
        Some(b'&' | b'*' | b'!') => return Err("anchors, aliases and tags are not supported".into()),
        _ => {}
    }
    Ok(match s {
        "" | "~" | "null" | "Null" | "NULL" => Value::Null,
        "true" | "True" | "TRUE" => Value::Bool(true),
        "false" | "False" | "FALSE" => Value::Bool(false),
        _ => {
            let digits = s.strip_prefix(['-', '+']).unwrap_or(s);
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) && !(digits.len() > 1 && digits.starts_with('0')) {
                s.parse::<i64>().map(Value::from).unwrap_or_else(|_| Value::String(s.to_owned()))
            } else if let Some(oct) = s.strip_prefix("0o") {
                i64::from_str_radix(oct, 8).map(Value::from).unwrap_or_else(|_| Value::String(s.to_owned()))
            } else {
                Value::String(s.to_owned())
            }
        }
    })
}

/// Splits a flow collection's inside at commas outside quotes and brackets.
fn split_flow(s: &str) -> Result<Vec<&str>, String> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0i32, 0);
    let (mut single, mut double) = (false, false);
    let b = s.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        match c {
            b'\'' if !double => single = !single,
            b'"' if !single && (i == 0 || b[i - 1] != b'\\') => double = !double,
            b'[' | b'{' if !single && !double => depth += 1,
            b']' | b'}' if !single && !double => depth -= 1,
            b',' if !single && !double && depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if single || double || depth != 0 {
        return Err("unterminated flow collection".into());
    }
    let last = s[start..].trim();
    if !last.is_empty() {
        out.push(last);
    }
    Ok(out)
}

fn double_quoted(s: &str) -> Result<String, String> {
    let inner = s.strip_prefix('"').and_then(|x| x.strip_suffix('"')).filter(|_| s.len() >= 2).ok_or("unterminated quote")?;
    let mut out = String::with_capacity(inner.len());
    let mut it = inner.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('/') => out.push('/'),
            Some(other) => return Err(format!("unsupported escape \\{other}")),
            None => return Err("unterminated escape".into()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::parse;
    use serde_json::json;

    #[test]
    fn block_mappings_and_sequences() {
        let v = parse("---\na: 1\nb:\n  c: x # comment\n  d: 'it''s'\nlist:\n- one\n- \"two: 2\"\n- k: v\n  k2: [a, \"b\", 3]\nnested:\n  - - x\n    - y\nempty: {}\nnone:\n").unwrap();
        assert_eq!(
            v,
            json!({"a": 1, "b": {"c": "x", "d": "it's"}, "list": ["one", "two: 2", {"k": "v", "k2": ["a", "b", 3]}], "nested": [["x", "y"]], "empty": {}, "none": null})
        );
    }

    #[test]
    fn strings_that_look_like_other_things() {
        let v = parse("name: \"0\"\nmode: \"755\"\nuuid: GPU-4a1b\nver: 580.95.05\nlink: ../card1::/dev/dri/by-path/pci-0000:21:00.0-card\nurl: a#b\nzero: 0755\n").unwrap();
        assert_eq!(v["name"], "0");
        assert_eq!(v["mode"], "755");
        assert_eq!(v["uuid"], "GPU-4a1b");
        assert_eq!(v["ver"], "580.95.05");
        assert_eq!(v["link"], "../card1::/dev/dri/by-path/pci-0000:21:00.0-card");
        assert_eq!(v["url"], "a#b");
        assert_eq!(v["zero"], "0755");
    }

    #[test]
    fn what_it_refuses() {
        assert!(parse("a: |\n  text\n").is_err());
        assert!(parse("a: &x 1\n").is_err());
        assert!(parse("a: 1\n---\nb: 2\n").is_err());
        assert!(parse("a: 1\na: 2\n").is_err());
        assert!(parse("a:\n    b: 1\n  c: 2\n").is_err());
        assert!(parse("just words").is_err());
    }
}
