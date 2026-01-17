# Engineering Team Guidelines

## Code Review Process

### Requirements
All code changes must be reviewed before merging to main:

1. **Small Changes** (< 100 lines): 1 reviewer required
2. **Medium Changes** (100-500 lines): 2 reviewers required
3. **Large Changes** (> 500 lines): 2 reviewers + tech lead approval

### Review Timeline
- Reviewers should respond within 24 hours
- Authors should address feedback within 48 hours
- If blocked, escalate to tech lead

### What to Look For
- Code correctness and logic errors
- Security vulnerabilities
- Performance implications
- Test coverage (minimum 80%)
- Documentation updates

## Git Workflow

### Branch Naming
```
feature/TICKET-123-description
bugfix/TICKET-456-description
hotfix/critical-issue-description
```

### Commit Messages
Follow conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `refactor:` for code refactoring
- `test:` for test additions

### Pull Request Template
All PRs must include:
- Description of changes
- Link to ticket/issue
- Testing steps
- Screenshots (for UI changes)

## Development Environment

### Required Tools
- Docker Desktop
- Node.js v18+
- Python 3.11+
- VS Code (recommended)

### Setup Steps
1. Clone the repository
2. Copy `.env.example` to `.env`
3. Run `docker-compose up -d`
4. Run `npm install` and `pip install -r requirements.txt`
5. Run `npm run dev` to start development server

### Database Access
- Development: localhost:5432
- Staging: Contact DevOps for credentials
- Production: Read-only access via VPN

## Deployment Process

### Environments
1. **Development**: Auto-deploys on push to `develop`
2. **Staging**: Manual deploy from `staging` branch
3. **Production**: Requires release tag and approval

### Release Checklist
- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Stakeholder notification sent

### Rollback Procedure
If issues are detected in production:
1. Alert #incidents Slack channel
2. Create incident ticket
3. Roll back to previous version
4. Investigate root cause
5. Post incident review within 48 hours

## On-Call Responsibilities

### Schedule
- Rotations are weekly, Monday 9 AM to Monday 9 AM
- Schedule posted in PagerDuty
- Swap requests via #oncall-swaps Slack channel

### Response Times
- **P1 (Critical)**: Respond within 15 minutes
- **P2 (High)**: Respond within 1 hour
- **P3 (Medium)**: Respond within 4 hours
- **P4 (Low)**: Next business day

### Escalation Path
1. On-call engineer
2. Secondary on-call
3. Team lead
4. Engineering manager
5. VP of Engineering

## Security Best Practices

### API Security
- All endpoints require authentication
- Use JWT tokens with short expiration
- Implement rate limiting
- Log all access attempts

### Secret Management
- Never commit secrets to git
- Use AWS Secrets Manager for production
- Rotate secrets quarterly
- Audit access logs monthly

### Dependency Management
- Run `npm audit` weekly
- Update dependencies monthly
- Critical vulnerabilities: patch within 24 hours

---

*Last updated: January 2026*
*Owner: Engineering Team*
