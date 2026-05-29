# Development Guidelines for AI CLI

## Introduction
This document outlines the guidelines for contributing to the AI CLI project. It includes instructions for setting up your development environment, coding standards, and best practices to follow while working on the project.

## Setting Up Your Development Environment

1. **Clone the Repository**
   Start by cloning the repository to your local machine:
   ```
   git clone https://github.com/yourusername/ai_cli.git
   cd ai_cli
   ```

2. **Create a Virtual Environment**
   It is recommended to use a virtual environment to manage dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   Install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the root directory based on the `.env.example` file. Make sure to set the necessary API keys and other environment variables required for the application to function properly.

## Coding Standards

- **Code Style**: Follow PEP 8 guidelines for Python code. Use tools like `flake8` or `black` for linting and formatting.
- **Documentation**: Ensure that all functions and classes are well-documented. Use docstrings to describe the purpose, parameters, and return values.
- **Testing**: Write unit tests for any new features or bug fixes. Place your tests in the `tests` directory. Run the comprehensive test suite using `poetry run pytest tests/test_enhanced.py -v` (and with coverage using `poetry run pytest --cov=src/ai_cli --cov-report=term-missing`). Ensure all tests pass and coverage is maintained or improved.

## Contributing

1. **Branching**: Create a new branch for your feature or bug fix:
   ```
   git checkout -b feature/your-feature-name
   ```

2. **Commit Changes**: Make your changes and commit them with a clear message:
   ```
   git commit -m "Add feature: description of the feature"
   ```

3. **Push Changes**: Push your branch to the remote repository:
   ```
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request**: Go to the GitHub repository and create a pull request from your branch to the main branch. Provide a description of the changes and any relevant information.

## Best Practices

- Keep your commits small and focused on a single task.
- Regularly pull changes from the main branch to keep your branch up to date.
- Review pull requests from others and provide constructive feedback.

## Conclusion
Thank you for contributing to the AI CLI project! Your efforts help improve the tool and make it more useful for everyone. If you have any questions or need assistance, feel free to reach out to the project maintainers.