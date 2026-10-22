export const MAX_SOURCE_BYTES = 25_000;

export function reviewSource(source) {
  const findings = [];
  const lines = source.split(/\r?\n/);
  const patterns = [
    {
      expression: /\b(?:eval|exec)\s*\(/,
      severity: "high",
      title: "Dynamic code execution",
      explanation: "eval() and exec() execute Python code constructed at runtime. If input is untrusted, this may allow arbitrary code execution.",
      recommendation: "Avoid dynamic evaluation. Parse the allowed input format or map accepted operations explicitly."
    },
    {
      expression: /^\s*except\s*:/,
      severity: "medium",
      title: "Bare except catches every exception",
      explanation: "This can hide unexpected failures and intercept control-flow exceptions.",
      recommendation: "Catch only the specific exception types this block can handle."
    },
    {
      expression: /\b(?:subprocess\.)?(?:run|Popen|call)\s*\([^\n]*\bshell\s*=\s*True/,
      severity: "high",
      title: "Subprocess call uses shell=True",
      explanation: "Building shell commands with untrusted values can enable command injection.",
      recommendation: "Pass an argument list with shell=False and validate user-controlled arguments."
    }
  ];

  lines.forEach((text, index) => {
    if (/^\s*#/.test(text)) return;
    for (const pattern of patterns) {
      if (pattern.expression.test(text)) {
        findings.push({
          severity: pattern.severity,
          title: pattern.title,
          line: index + 1,
          explanation: pattern.explanation,
          recommendation: pattern.recommendation
        });
      }
    }
  });

  return findings;
}
