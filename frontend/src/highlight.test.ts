import { describe, expect, it } from "vitest";
import { splitByQuote } from "./highlight";

describe("splitByQuote", () => {
  it("marks an exact quote", () => {
    expect(splitByQuote("Refunds take 14 days.", "take 14 days")).toEqual([
      { text: "Refunds ", match: false },
      { text: "take 14 days", match: true },
      { text: ".", match: false },
    ]);
  });

  it("tolerates different case and whitespace", () => {
    const segments = splitByQuote("Refunds take\n14   days.", "TAKE 14 days");
    expect(segments.find((s) => s.match)?.text).toBe("take\n14   days");
  });

  it("treats regex characters in the quote literally", () => {
    const segments = splitByQuote("Price (net) is $5.00 [EUR]", "(net) is $5.00 [EUR]");
    expect(segments.find((s) => s.match)?.text).toBe("(net) is $5.00 [EUR]");
  });

  it("returns the text unmarked when the quote is absent", () => {
    expect(splitByQuote("abc", "xyz")).toEqual([{ text: "abc", match: false }]);
  });
});
