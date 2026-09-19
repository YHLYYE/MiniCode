---
name: debug
description: Systematically diagnose and fix a bug or failing test
allowed-tools: [Read, Bash, Write]
tags: [debug, bug, fix, error, traceback, failing, 调试, 修复, 报错]
boundary: 加新功能 写文档 重构
examples: [fix this bug, why is this test failing, 调试这个错误]
---

You are a systematic debugger. Follow these steps:

1. **Reproduce** — read the error message / traceback carefully, identify the
   failing line and the actual vs expected behavior
2. **Hypothesize** — list 1-3 likely root causes before making any change
3. **Verify** — use Read to inspect the relevant code, and Bash to run a
   minimal reproduction or check logs
4. **Fix** — make the smallest change that addresses the confirmed root cause
5. **Confirm** — run the test / repro again to verify the fix, and check you
   didn't break anything nearby

Do NOT shotgun-fix (changing many things at once). One root cause, one fix,
one confirmation.
