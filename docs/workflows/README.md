# GitHub Actions Workflows Documentation

This directory contains documentation for the recommended GitHub Actions workflows for the protein-diffusion-design-lab project.

**Note**: As per Terragon security policy, actual workflow files must be manually created by maintainers. This documentation provides templates and implementation guidance.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Run tests, linting, and security checks on every push and PR

**Triggers**:
- Push to main branch
- Pull requests to main branch
- Manual dispatch

**Jobs**:
- Code quality checks (black, isort, flake8, mypy)
- Security scanning (bandit, safety, GitGuardian)
- Unit tests with coverage reporting
- Integration tests
- GPU tests (conditional)

**Implementation Location**: `.github/workflows/ci.yml`

Refer to: [ci-workflow-template.md](./ci-workflow-template.md)

### 2. Release Management (`release.yml`)

**Purpose**: Automated release creation and PyPI publishing

**Triggers**:
- Version tags (v*.*.*)
- Manual dispatch

**Jobs**:
- Build distribution packages
- Run full test suite
- Publish to PyPI (with approval gate)
- Create GitHub release with changelog
- Update documentation

**Implementation Location**: `.github/workflows/release.yml`

Refer to: [release-workflow-template.md](./release-workflow-template.md)

### 3. Security Scanning (`security.yml`)

**Purpose**: Comprehensive security analysis and vulnerability detection

**Triggers**:
- Schedule (daily)
- Push to main branch
- Manual dispatch

**Jobs**:
- Dependency vulnerability scanning
- Container security analysis
- SAST (Static Application Security Testing)
- License compliance checking
- SBOM generation

**Implementation Location**: `.github/workflows/security.yml`

Refer to: [security-workflow-template.md](./security-workflow-template.md)

### 4. Documentation (`docs.yml`)

**Purpose**: Build and deploy documentation

**Triggers**:
- Push to main branch
- Pull requests affecting docs/
- Manual dispatch

**Jobs**:
- Build Sphinx documentation
- Deploy to GitHub Pages
- Link checking
- Documentation quality checks

**Implementation Location**: `.github/workflows/docs.yml`

Refer to: [docs-workflow-template.md](./docs-workflow-template.md)

## Workflow Integration Requirements

### External Services

1. **CodeCov**: Code coverage reporting
   - Setup token in repository secrets
   - Configure coverage thresholds

2. **Dependabot**: Automated dependency updates
   - Configure `.github/dependabot.yml`

3. **GitGuardian**: Secret scanning
   - Validate API key configuration

### Repository Secrets

Required secrets for workflow execution:

```yaml
PYPI_API_TOKEN: # PyPI publishing token
CODECOV_TOKEN: # Coverage reporting token
GITGUARDIAN_API_KEY: # Secret scanning API key
```

### Branch Protection Rules

Recommended branch protection for `main`:

- Require pull request reviews (1+ approvers)
- Require status checks (CI must pass)
- Require up-to-date branches
- Include administrators in restrictions
- Allow force pushes: false
- Allow deletions: false

## Manual Setup Steps

1. **Create workflow files** from templates in individual docs
2. **Configure repository secrets** listed above
3. **Enable GitHub Pages** for documentation deployment
4. **Configure branch protection** rules
5. **Setup Dependabot** configuration
6. **Test workflows** with sample PR

## Monitoring and Maintenance

- Review workflow execution weekly
- Update action versions quarterly
- Monitor security scan results daily
- Adjust coverage thresholds based on project evolution

## Troubleshooting

Common issues and solutions documented in individual workflow templates.

For questions or issues, refer to the [CONTRIBUTING.md](../../CONTRIBUTING.md) guide.