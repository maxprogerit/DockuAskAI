import type { Citation } from "./types";

/** "p. 4" for PDFs; heading path / paragraph number for DOCX (which has no pages). */
export function citationLocation(c: Citation): string {
  if (c.page) return `p.${c.page}`;
  const parts: string[] = [];
  if (c.section) parts.push(c.section.split(">").pop()!.trim());
  if (c.para_index !== null) parts.push(`¶${c.para_index}`);
  return parts.join(" · ");
}

/** "[1] contract.pdf · p.4" */
export function citationLabel(c: Citation, index: number): string {
  const where = citationLocation(c);
  return `[${index + 1}] ${c.filename}${where ? ` · ${where}` : ""}`;
}
