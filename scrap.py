def lcs(x, y):
    """Longest Common Subsequence (LCS) length."""
    m, n = len(x), len(y)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[m][n]

def rouge_l(candidate, reference):
    """
    Calculate ROUGE-L score between candidate and reference sentences.
    """
    # Tokenize by whitespace
    candidate = candidate.lower()
    reference = reference.lower()
    cand_tokens = candidate.split()
    ref_tokens = reference.split()

    lcs_length = lcs(cand_tokens, ref_tokens)

    prec = lcs_length / len(cand_tokens) if cand_tokens else 0.0
    rec = lcs_length / len(ref_tokens) if ref_tokens else 0.0

    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = (2 * prec * rec) / (prec + rec)

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

# Example usage
candidate = "Her name is Chun Li"
reference = "The author's full name is Hsiao Yun-Hwa"

score = rouge_l(candidate, reference)
print("ROUGE-L:", score)
