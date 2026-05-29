from __future__ import annotations
import re
from typing import Dict, List, Optional
from ai_cli.core.exceptions import PromptValidationError


class PromptCorrector:
    """
    Intelligent correction and sanitization gateway for user prompts.

    This class provides heuristic and rule-based corrections for common user formatting
    errors, typos, control characters, and punctuation mistakes before prompts are
    sent to downstream LLM providers.

    Features:
    1. Whitespace normalization: collapses multiple consecutive spaces.
    2. Punctuation spacing correction: eliminates spaces before punctuation (e.g. "word ?")
       and ensures proper spacing after punctuation.
    3. Common typo correction: replaces common misspelled words (e.g., "teh" -> "the").
    4. Control/NUL byte stripping.
    5. Parentheses/Brackets/Quotes balancing: appends missing closing symbols to ensure well-formed structures.
    """

    DEFAULT_TYPO_MAP: Dict[str, str] = {
        "teh": "the",
        "dont": "don't",
        "cant": "can't",
        "wont": "won't",
        "recieve": "receive",
        "seperate": "separate",
        "occurance": "occurrence",
        "definately": "definitely",
        "goverment": "government",
        "wierd": "weird",
    }

    def __init__(
        self,
        typo_map: Optional[Dict[str, str]] = None,
        collapse_spaces: bool = True,
        fix_punctuation: bool = True,
        balance_brackets: bool = True,
        clean_control_chars: bool = True,
    ) -> None:
        self.typo_map = typo_map if typo_map is not None else self.DEFAULT_TYPO_MAP
        self.collapse_spaces = collapse_spaces
        self.fix_punctuation = fix_punctuation
        self.balance_brackets = balance_brackets
        self.clean_control_chars = clean_control_chars

        # Compile typo regexes for faster matching (case-insensitive word boundary)
        self._typo_regexes = {
            re.compile(rf"\b{re.escape(typo)}\b", re.IGNORECASE): replacement
            for typo, replacement in self.typo_map.items()
        }

    def correct(self, prompt: str) -> str:
        """
        Apply enabled correction rules to the prompt and return the corrected string.

        Args:
            prompt (str): The raw user prompt to correct.

        Returns:
            str: The corrected and sanitized prompt.

        Raises:
            PromptValidationError: If the input is not a string.
        """
        if not isinstance(prompt, str):
            raise PromptValidationError("prompt must be string")

        corrected = prompt

        # 1. Clean control characters and NUL bytes
        if self.clean_control_chars:
            corrected = corrected.replace("\x00", "")
            # Remove control characters except tab and newline
            corrected = "".join(
                ch for ch in corrected if ch in ("\n", "\t") or ord(ch) >= 32
            )

        # 2. Normalise whitespace
        if self.collapse_spaces:
            # Collapse multiple spaces/tabs on each line but preserve newlines
            lines = []
            for line in corrected.splitlines():
                collapsed_line = re.sub(r"[ \t]+", " ", line)
                lines.append(collapsed_line)
            # Reconstruct with original line breaks, stripping outer whitespace
            corrected = "\n".join(lines).strip()
        else:
            corrected = corrected.strip()

        if not corrected:
            return ""

        # 3. Correct common typos (word boundaries, preserving casing)
        for pattern, replacement in self._typo_regexes.items():
            def repl_func(match: re.Match) -> str:
                word = match.group(0)
                if word.isupper():
                    return replacement.upper()
                if word[0].isupper():
                    return replacement.capitalize()
                return replacement

            corrected = pattern.sub(repl_func, corrected)

        # 4. Fix punctuation spacing
        if self.fix_punctuation:
            # Remove spaces before punctuation: ?, !, ., ,, ;, :
            corrected = re.sub(r"\s+([?!.,;:])", r"\1", corrected)
            # Ensure space after punctuation if followed by letters or digits
            corrected = re.sub(r"([?!.,;:])([A-Za-z0-9])", r"\1 \2", corrected)

        # 5. Balance brackets and quotes
        if self.balance_brackets:
            corrected = self._balance_brackets_and_quotes(corrected)

        return corrected

    def _balance_brackets_and_quotes(self, text: str) -> str:
        """Append missing closing brackets/parentheses and ensure quotes are balanced."""
        stack: List[str] = []
        bracket_pairs = {")": "(", "]": "[", "}": "{"}
        open_brackets = {"(", "[", "{"}

        sq_count = 0
        dq_count = 0

        filtered_chars = []
        for idx, char in enumerate(text):
            if char in open_brackets:
                stack.append(char)
                filtered_chars.append(char)
            elif char in bracket_pairs:
                target_open = bracket_pairs[char]
                if target_open in stack:
                    while stack:
                        popped = stack.pop()
                        if popped == target_open:
                            break
                    filtered_chars.append(char)
                else:
                    # Omit unmatched closing bracket
                    pass
            elif char == "'":
                # Check if it's an apostrophe (surrounded by letters/digits on both sides)
                is_apostrophe = False
                if idx > 0 and idx < len(text) - 1:
                    prev_char = text[idx - 1]
                    next_char = text[idx + 1]
                    if prev_char.isalnum() and next_char.isalnum():
                        is_apostrophe = True
                if not is_apostrophe:
                    sq_count += 1
                filtered_chars.append(char)
            elif char == '"':
                dq_count += 1
                filtered_chars.append(char)
            else:
                filtered_chars.append(char)

        # Reconstruct text without mismatched closing brackets
        result = "".join(filtered_chars)

        # Append missing closing brackets in reverse order
        appendices = []
        reverse_pairs = {"(": ")", "[": "]", "{": "}"}
        while stack:
            popped = stack.pop()
            appendices.append(reverse_pairs[popped])

        result = result + "".join(appendices)

        # Append quotes if they are odd in count
        if sq_count % 2 != 0:
            result += "'"
        if dq_count % 2 != 0:
            result += '"'

        return result


# Global convenience instance
prompt_corrector = PromptCorrector()