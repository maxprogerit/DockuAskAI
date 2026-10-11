export interface Segment {
  text: string;
  match: boolean;
}

/**
 * Splits `text` around the first occurrence of `quote` so the UI can <mark> it.
 * The backend verifies the quote against the chunk after normalizing case/whitespace, so an exact
 * indexOf can miss; matching word by word with flexible whitespace and case covers that. If there
 * is still no match the whole text is returned unmarked - never a wrong highlight.
 */
export function splitByQuote(text: string, quote: string): Segment[] {
  const words = quote.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return [{ text, match: false }];
  const escaped = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const found = new RegExp(escaped.join("\\s+"), "i").exec(text);
  if (!found) return [{ text, match: false }];
  const start = found.index;
  const end = start + found[0].length;
  return [
    { text: text.slice(0, start), match: false },
    { text: text.slice(start, end), match: true },
    { text: text.slice(end), match: false },
  ].filter((s) => s.text.length > 0);
}
