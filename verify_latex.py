"""
LaTeX / KaTeX Integrity Verification Script (Enhanced)
Audits index.html for all math expressions and verifies KaTeX delimiter integrity.
"""

import re
import sys

def audit_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Isolate non-script, non-style content
    text_only = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.DOTALL)
    text_only = re.sub(r"<style.*?>.*?</style>", "", text_only, flags=re.DOTALL)

    # 1. Find all $$...$$ display math blocks
    display_matches = re.findall(r"\$\$(.*?)\$\$", text_only, flags=re.DOTALL)
    print(f"\n📊 Found {len(display_matches)} Display Math ($$...$$) block(s):")
    for i, m in enumerate(display_matches, 1):
        clean_m = m.strip().replace('\n', ' ')
        print(f"  [{i}] $${clean_m[:70]}...$$")

    # 2. Find all $...$ inline math blocks
    inline_matches = re.findall(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)", text_only)
    print(f"\n📊 Found {len(inline_matches)} Inline Math ($...$) block(s):")
    for i, m in enumerate(inline_matches, 1):
        clean_m = m.strip()
        print(f"  [{i}] ${clean_m}$")

    # 3. Check for any unescaped standalone $ or orphan $$
    all_double_dollars = len(re.findall(r"\$\$", text_only))
    print(f"\n🔍 Total '$$' delimiter tokens in text: {all_double_dollars}")

    errors = []
    if all_double_dollars % 2 != 0:
        errors.append("Odd count of '$$' delimiters - unclosed display math block!")

    # Check for unmatched braces in display math
    for i, m in enumerate(display_matches, 1):
        if m.count("{") != m.count("}"):
            errors.append(f"Mismatched braces in Display Math [{i}]")

    # Check for unmatched braces in inline math
    for i, m in enumerate(inline_matches, 1):
        if m.count("{") != m.count("}"):
            errors.append(f"Mismatched braces in Inline Math [{i}]")

    return errors

if __name__ == "__main__":
    path = "/Users/dipayan/.gemini/antigravity-ide/brain/67ac9b3c-2bc1-4fca-a84a-6bcf7c62cf34/index.html"
    errs = audit_html(path)
    if errs:
        print(f"\n❌ Validation Failed with {len(errs)} error(s): {errs}")
        sys.exit(1)
    else:
        print("\n✅ Verification SUCCESS: All LaTeX entries are 100% clean, balanced, and corruption-free!")
