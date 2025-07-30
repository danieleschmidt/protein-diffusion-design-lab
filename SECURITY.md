# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in protein-diffusion-design-lab, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please:

1. **Email us**: Send a detailed report to security@terragonlabs.com
2. **Include**: 
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if available)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 5 business days  
- **Fix Timeline**: Varies by severity (see below)
- **Public Disclosure**: After fix is released

### Severity Guidelines

| Severity | Description | Response Time |
|----------|-------------|---------------|
| **Critical** | Remote code execution, privilege escalation | 24-48 hours |
| **High** | Data exposure, authentication bypass | 3-7 days |
| **Medium** | Information disclosure, DoS attacks | 1-2 weeks |
| **Low** | Configuration issues, minor information leaks | 2-4 weeks |

## Security Best Practices

### For Users

- **Keep Updated**: Always use the latest version
- **Secure Configs**: Follow configuration security guidelines
- **Validate Inputs**: Never trust user-provided protein sequences without validation
- **Model Security**: Only load models from trusted sources
- **Environment**: Use virtual environments and dependency pinning

### For Contributors

- **Dependency Scanning**: Run `safety check` before committing
- **Secret Detection**: Use pre-commit hooks to prevent secret commits
- **Code Review**: Security-focused review for all changes
- **Testing**: Include security test cases

## Known Security Considerations

### Model Loading
- **Risk**: Malicious pickle files in model checkpoints
- **Mitigation**: Validate model sources, use safe loading methods

### Input Processing  
- **Risk**: Malformed protein sequences causing buffer overflows
- **Mitigation**: Input validation and sanitization

### Dependency Vulnerabilities
- **Risk**: Vulnerable ML/scientific computing dependencies
- **Mitigation**: Regular dependency updates, vulnerability scanning

### GPU Memory
- **Risk**: GPU memory leaks or unauthorized access
- **Mitigation**: Proper resource cleanup, memory monitoring

## Security Tools Integration

We use the following security tools:

- **Bandit**: Python AST security analyzer
- **Safety**: Python dependency vulnerability scanner  
- **GitGuardian**: Secret detection in commits
- **GitHub Security**: Dependabot security updates
- **Pre-commit**: Automated security checks

## Compliance

This project follows:

- **OWASP Top 10**: Web application security risks
- **CWE**: Common Weakness Enumeration guidelines
- **NIST**: Cybersecurity framework principles
- **Scientific Software**: Domain-specific security practices

## Contact

For security-related questions or concerns:

- **Security Team**: security@terragonlabs.com
- **General Contact**: support@terragonlabs.com
- **Bug Bounty**: Currently not available

---

**Remember**: Security is everyone's responsibility. Thank you for helping keep protein-diffusion-design-lab safe and secure!