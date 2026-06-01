#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
GOLD='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

echo_error() {
    echo -e "${RED}${BOLD}[ERROR]${RESET} $1"
}
echo_success() {
    echo -e "${GREEN}${BOLD}[SUCCESS]${RESET} $1"
}
echo_info() {
    echo -e "${BLUE}${BOLD}[INFO]${RESET} $1"
}
echo_warning() {
    echo -e "${GOLD}${BOLD}[WARNING]${RESET} $1"
}

# Function to display help
show_help() {
    echo -e "${GOLD}${BOLD}Usage:${RESET} ./install.sh [options]"
    echo -e "\nOptions:"
    echo -e "  -h, --help        Show this help message and exit"
    echo -e "  -u, --upgrade     Upgrade pip and install/upgrade all dependencies"
    echo -e "\nExample:\n  ./install.sh --upgrade"
}

UPGRADE=false

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -u|--upgrade)
            UPGRADE=true
            shift
            ;;
        *)
            echo_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Make activate script executable
chmod +x activate.sh

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo_error "Virtual environment not found. Please run the setup script first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip if requested
if [ "$UPGRADE" = true ]; then
    echo_info "Upgrading pip..."
    pip install --upgrade pip
fi

# Install dependencies
echo_info "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo_success "Dependencies installed successfully."
    echo_info "Virtual environment is ready to use."
    echo_warning "Run './activate.sh' to activate the environment."
else
    echo_error "Failed to install dependencies."
    exit 1
fi
