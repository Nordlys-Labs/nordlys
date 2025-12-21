# Contributing to Nordlys

Thank you for your interest in contributing to Nordlys! This guide will help you get started with contributing to the project.

## Getting Started

### Prerequisites

- Node.js 18+ and Bun
- Go 1.21+
- Python 3.11+
- Docker and Docker Compose

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Nordlys-Labs/nordlys.git
   cd nordlys
   ```

2. Set up the frontend (nordlys-frontend):
   ```bash
   cd nordlys-frontend
   bun install
   ```

3. Set up the backend (nordlys-backend):
   ```bash
   cd nordlys-backend
   go mod download
   ```

4. Set up the AI service (nordlys-py):
   ```bash
   cd nordlys-py
   pip install -r requirements.txt
   ```

5. Start the development environment:
   ```bash
   docker-compose up -d
   ```

## Development Workflow

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards below

3. Test your changes thoroughly

4. Commit your changes with a clear message

5. Push your branch and create a pull request

### Coding Standards

#### Frontend (TypeScript/React)
- Use TypeScript for all new code
- Follow the existing ESLint and Prettier configuration
- Use meaningful component and variable names
- Write tests for new components and utilities

#### Backend (Go)
- Follow Go conventions and best practices
- Use proper error handling
- Write unit tests for new functions
- Document exported functions and types

#### AI Service (Python)
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for modules, classes, and functions
- Include unit tests for new functionality

### Testing

- Run tests before submitting a PR:
  ```bash
  # Frontend
  cd nordlys-frontend && bun test
  
  # Backend
  cd nordlys-backend && go test ./...
  
  # AI Service
  cd nordlys-py && python -m pytest
  ```

### Pull Request Process

1. Ensure your code follows the project's coding standards
2. Update documentation if needed
3. Add tests for new functionality
4. Ensure all tests pass
5. Fill out the pull request template completely
6. Request review from maintainers

## Project Structure

```
nordlys/
├── nordlys-frontend/      # Next.js frontend application
├── nordlys-backend/       # Go backend service
├── nordlys-py/            # Python AI service
├── analysis/             # Analysis tools and scripts
├── monitoring/           # Monitoring and observability
└── docker-compose.yml    # Development environment
```

## Reporting Issues

- Use the GitHub issue tracker
- Provide a clear description of the problem
- Include steps to reproduce the issue
- Add relevant logs or error messages
- Specify your environment (OS, versions, etc.)

## Feature Requests

- Open an issue with the "enhancement" label
- Clearly describe the proposed feature
- Explain the use case and benefits
- Discuss implementation approach if applicable

## Questions and Support

- Check existing issues and documentation first
- Open a discussion for general questions
- Tag maintainers for urgent issues

## License

By contributing to Nordlys, you agree that your contributions will be licensed under the same license as the project.
