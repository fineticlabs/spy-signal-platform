Review all code changes in the given scope. If no scope is provided, review all uncommitted changes (`git diff` + `git diff --cached`).

For each change, evaluate:

1. **Correctness**: Does the logic do what it claims? Are there off-by-one errors, wrong comparisons, or missing edge cases?
2. **Edge Cases**: What happens with empty input, NaN values, zero volume, single-bar data, timezone-naive timestamps?
3. **Security**: Any hardcoded secrets, SQL injection, command injection, or unsafe deserialization?
4. **Testing**: Are there tests for the new/changed code? Do they cover the happy path AND edge cases?
5. **Performance**: Any O(n^2) loops on large datasets? Unnecessary copies of DataFrames? Missing vectorization?
6. **Standards Compliance**: Type hints on all functions? Structlog (not print)? Decimal for prices? Pydantic models?
7. **No-Lookahead**: In backtest code, are indicators shifted properly? Is bar close used only after bar completes?

Be specific — reference file paths and line numbers. Don't just say "looks good". If something is wrong, say what and how to fix it.

Scope to review: $ARGUMENTS
