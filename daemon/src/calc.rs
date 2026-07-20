//! Deterministic arithmetic for the CALC_NEEDED fault. The answer model is
//! told never to do math on remembered numbers or dates in its head;
//! answer-time synthesis working "by the model's own reasoning" is exactly
//! the kind of result that degrades first on long transcripts, so the
//! arithmetic goes through here and comes back exact.
//!
//! Two forms:
//!   plain arithmetic     "1800 + 200", "(62 - 50) * 1000"
//!   date shifts          "October 14 + 7 days", "March 3 2027 - 2 weeks"

const MONTHS: [&str; 12] = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
];

pub fn eval(raw: &str) -> Result<String, String> {
    let expr = raw.trim().trim_matches(['`', '"', '\'']).trim();
    if expr.is_empty() {
        return Err("empty expression".into());
    }
    if let Some(out) = date_shift(expr) {
        return Ok(out);
    }
    arith(&normalize(expr)).map(format_number)
}

/// Models write "15 minus 12", "620 gigabytes - 140 gigabytes" and
/// "15 (current headcount) - 12 (budgeted)". A strict parser turned each
/// of those into a wasted action-loop round (observed burning three of
/// four rounds on one turn), so word operators are translated,
/// letter-bearing parentheticals are dropped, and stray unit words are
/// stripped. Parentheses containing only math survive untouched.
fn normalize(expr: &str) -> String {
    let mut s = String::with_capacity(expr.len());
    let mut chars = expr.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '(' {
            let mut inner = String::new();
            let mut depth = 1;
            for d in chars.by_ref() {
                if d == '(' {
                    depth += 1;
                } else if d == ')' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                inner.push(d);
            }
            if inner.chars().any(|d| d.is_alphabetic()) {
                continue; // annotation like "(current headcount)": drop it
            }
            s.push('(');
            s.push_str(&inner);
            s.push(')');
        } else {
            s.push(c);
        }
    }
    // Word operators, then strip any remaining letter runs (unit words).
    let lowered = s.to_lowercase();
    let mut out = String::with_capacity(lowered.len());
    for tok in lowered.split_whitespace() {
        let t = tok.trim_matches(|c: char| c == ',' && false);
        match t {
            "plus" | "add" | "added" => out.push_str(" + "),
            "minus" | "subtract" | "less" => out.push_str(" - "),
            "times" | "multiplied" | "x" => out.push_str(" * "),
            "divided" | "over" => out.push_str(" / "),
            "by" => {} // "divided by", "multiplied by": the operator already landed
            _ if t.chars().all(|c| c.is_alphabetic()) => {} // unit word: drop
            _ => {
                out.push(' ');
                out.push_str(t);
            }
        }
    }
    out.trim().to_string()
}

// --- dates -----------------------------------------------------------------

/// "<Month> <day>[ <year>] +|- <n> day(s)|week(s)" -> "<Month> <day>[ <year>]"
fn date_shift(expr: &str) -> Option<String> {
    let cleaned = expr.replace(',', " ");
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    if words.len() < 5 {
        return None;
    }
    let first = words[0].to_lowercase();
    if first.len() < 3 {
        return None;
    }
    let month = MONTHS.iter().position(|m| m.starts_with(&first))? as u32 + 1;
    let day: u32 = words[1].trim_end_matches(['s', 't', 'n', 'r', 'd', 'h']).parse().ok()?;
    let (year, had_year, mut i) = match words[2].parse::<i64>() {
        Ok(y) if y > 1900 => (y, true, 3),
        _ => (current_year(), false, 2),
    };
    let sign: i64 = match *words.get(i)? {
        "+" | "plus" => 1,
        "-" | "minus" => -1,
        _ => return None,
    };
    i += 1;
    let n: i64 = words.get(i)?.parse().ok()?;
    i += 1;
    let unit = words.get(i)?.to_lowercase();
    let delta = if unit.starts_with("week") {
        n * 7
    } else if unit.starts_with("day") {
        n
    } else {
        return None;
    };
    let shifted = crate::bedrock::civil_from_days(crate::bedrock::days_from_civil(year, month, day) + sign * delta);
    let name = capitalize(MONTHS[(shifted.1 - 1) as usize]);
    Some(if had_year || shifted.0 != current_year() {
        format!("{name} {} {}", shifted.2, shifted.0)
    } else {
        format!("{name} {}", shifted.2)
    })
}

fn current_year() -> i64 {
    let days = (crate::state::now_ms() / 86_400_000) as i64;
    crate::bedrock::civil_from_days(days).0
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

// --- arithmetic ------------------------------------------------------------

/// Recursive descent over + - * / ( ) and numbers ("$1,800" reads as 1800).
fn arith(expr: &str) -> Result<f64, String> {
    let tokens = tokenize(expr)?;
    let mut pos = 0;
    let v = parse_expr(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err(format!("unexpected '{:?}'", tokens[pos]));
    }
    Ok(v)
}

#[derive(Debug, PartialEq)]
enum Tok {
    Num(f64),
    Op(char),
}

fn tokenize(expr: &str) -> Result<Vec<Tok>, String> {
    let mut out = Vec::new();
    let cleaned = expr.replace(['$', ','], "");
    let mut chars = cleaned.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' => {
                chars.next();
            }
            '+' | '-' | '*' | '/' | '(' | ')' | 'x' => {
                out.push(Tok::Op(if c == 'x' { '*' } else { c }));
                chars.next();
            }
            '0'..='9' | '.' => {
                let mut n = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() || d == '.' {
                        n.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                out.push(Tok::Num(n.parse().map_err(|_| format!("bad number '{n}'"))?));
            }
            other => return Err(format!("can't parse '{other}'")),
        }
    }
    if out.is_empty() {
        return Err("nothing to compute".into());
    }
    Ok(out)
}

fn parse_expr(t: &[Tok], pos: &mut usize) -> Result<f64, String> {
    let mut v = parse_term(t, pos)?;
    while let Some(Tok::Op(op @ ('+' | '-'))) = t.get(*pos) {
        let op = *op;
        *pos += 1;
        let rhs = parse_term(t, pos)?;
        v = if op == '+' { v + rhs } else { v - rhs };
    }
    Ok(v)
}

fn parse_term(t: &[Tok], pos: &mut usize) -> Result<f64, String> {
    let mut v = parse_factor(t, pos)?;
    while let Some(Tok::Op(op @ ('*' | '/'))) = t.get(*pos) {
        let op = *op;
        *pos += 1;
        let rhs = parse_factor(t, pos)?;
        if op == '/' && rhs == 0.0 {
            return Err("division by zero".into());
        }
        v = if op == '*' { v * rhs } else { v / rhs };
    }
    Ok(v)
}

fn parse_factor(t: &[Tok], pos: &mut usize) -> Result<f64, String> {
    match t.get(*pos) {
        Some(Tok::Num(n)) => {
            *pos += 1;
            Ok(*n)
        }
        Some(Tok::Op('-')) => {
            *pos += 1;
            Ok(-parse_factor(t, pos)?)
        }
        Some(Tok::Op('(')) => {
            *pos += 1;
            let v = parse_expr(t, pos)?;
            match t.get(*pos) {
                Some(Tok::Op(')')) => {
                    *pos += 1;
                    Ok(v)
                }
                _ => Err("missing )".into()),
            }
        }
        other => Err(format!("expected a number, got {other:?}")),
    }
}

fn format_number(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        assert_eq!(eval("1800 + 200").unwrap(), "2000");
        assert_eq!(eval("$1,800 + $200").unwrap(), "2000");
        assert_eq!(eval("(62 - 50) * 1000").unwrap(), "12000");
        assert_eq!(eval("10 / 4").unwrap(), "2.50");
        assert!(eval("1 / 0").is_err());
        assert!(eval("what").is_err());
    }

    #[test]
    fn model_shaped_expressions_from_the_probe_trace() {
        // The exact three expressions that each burned an action-loop round.
        assert_eq!(eval("15 (current headcount) - 12 (budgeted headcount)").unwrap(), "3");
        assert_eq!(eval("15 minus 12").unwrap(), "3");
        assert_eq!(eval("620 gigabytes - 140 gigabytes").unwrap(), "480");
        // And relatives seen in earlier journals.
        assert_eq!(eval("620 gigabytes - 140 gigabytes in gigabytes").unwrap(), "480");
        assert_eq!(eval("1800 plus 200").unwrap(), "2000");
        assert_eq!(eval("62000 divided by 2").unwrap(), "31000");
        // Numeric parentheses still work after annotation stripping.
        assert_eq!(eval("(3 + 4) * 2 (final)").unwrap(), "14");
        // Pure words still refuse rather than guessing.
        assert!(eval("current headcount minus budgeted headcount").is_err());
    }

    #[test]
    fn date_shifts() {
        assert_eq!(eval("October 14 + 7 days").unwrap(), "October 21");
        assert_eq!(eval("October 14 + 1 week").unwrap(), "October 21");
        assert_eq!(eval("March 3 2027 - 2 weeks").unwrap(), "February 17 2027");
        // Month rollover.
        assert_eq!(eval("October 28 + 7 days").unwrap(), "November 4");
    }
}
