# Nina_advanced_api_mcp
Inter face for AI agents to use your astrophotography setup using N.I.N.A
# NINA Model Context Provider (MCP)

A powerful interface for controlling N.I.N.A. (Nighttime Imaging 'N' Astronomy) software through its Advanced API. This Model Context Provider (MCP) enables AI agents to interact with NINA, providing seamless automation and integration for astrophotography.

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NINA](https://img.shields.io/badge/NINA-3.0+-green.svg)](https://nighttime-imaging.eu/)

</div>

## 🌟 Features

- **Complete Equipment Control**
  - Cameras (capture, cooling, settings)
  - Mounts (slewing, parking, tracking)
  - Focusers (movement, temperature compensation)
  - Filter Wheels (filter selection, info)
  - Domes (rotation, shutter control)
  - Rotators (movement, sync)

- **Advanced Imaging**
  - Dynamic exposure timeouts
  - Platesolving integration
  - Image streaming and format conversion
  - Automatic file saving
  - Image statistics and analysis

- **Smart Error Handling**
  - Comprehensive error recovery
  - Detailed logging
  - Connection management
  - Hardware state monitoring

- **AI Integration**
  - Natural language command processing
  - Contextual help system
  - Intelligent error responses
  - Automated decision making

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- NINA software with Advanced API plugin
- `uv` package manager
- AI agent with MCP support (e.g., Claude)

### Installation

1. **Install NINA Advanced API Plugin**
   ```bash
   # Download and install through NINA's plugin manager
   # Enable and configure in NINA settings
   ```

2. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/nina-mcp.git
   cd nina-mcp
   ```

3. **Set Environment Variables**
   ```bash
   # Create .env file
   NINA_HOST=your_nina_host
   NINA_PORT=1888
   LOG_LEVEL=INFO
   IMAGE_SAVE_DIR=~/Desktop/NINA_Images
   ```

### Configuration

#### MCP Server Setup
Add to your AI agent's MCP configuration:
```json
{
  "mcpServers": {
    "nina_advanced_mcp_api": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "fastmcp,fastapi,uvicorn,pydantic,aiohttp,requests,python-dotenv",
        "fastmcp",
        "run",
        "nina_advanced_mcp.py"
      ],
      "env": {
        "NINA_HOST": "NINA_IP",
        "NINA_PORT": "1888",
        "LOG_LEVEL": "INFO",
        "IMAGE_SAVE_DIR": "~/Desktop/NINA_Images"
      }
    }
  }
}
```

## 📚 Usage

### Basic Examples

```python
# Connect to NINA
result = await nina_connect({"host": "localhost", "port": 1888})

# Capture an Image
result = await nina_capture_image({
    "duration": 30,
    "download": True,
    "solve": True,
    "quality": 90,
    "size": "1920x1080"
})

# Control Mount
result = await nina_slew_mount({
    "ra": 12.5,  # Hours
    "dec": 45.0  # Degrees
})
```

### AI Agent Commands

```plaintext
- "Take a 30-second exposure of M31"
- "Connect all equipment and start cooling the camera to -10°C"
- "Start a sequence targeting NGC 7000"
- "Get the current equipment status"
```

## 🛠 Advanced Features

### Dynamic Timeouts
- Automatic timeout calculation based on:
  - Exposure duration
  - Platesolving requirements
  - Download settings
  - Operation type

### Error Recovery
- Automatic retry mechanisms
- Hardware state recovery
- Connection management
- Comprehensive error reporting

### Image Processing
- Format conversion
- Size adjustment
- Quality control
- Metadata preservation

## 📖 API Documentation

### Core Modules

#### Equipment Control
- Camera operations
- Mount control
- Focuser management
- Filter wheel control
- Dome automation
- Rotator functions

#### Imaging
- Capture configuration
- Image processing
- File management
- Statistics gathering

#### System
- Connection handling
- Status monitoring
- Error management
- Configuration

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 🐛 Bug Reports

Found a bug? Please open an issue with:
- Detailed description
- Steps to reproduce
- Expected vs actual behavior
- System information

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NINA](https://nighttime-imaging.eu/) - The core astronomy software
- [NINA Advanced API](https://bump.sh/christian-photo/doc/advanced-api) - API documentation
- Community contributors and testers

## 🔗 Related Projects

- [Touch'N'Stars](https://github.com/Touch-N-Stars/Touch-N-Stars) - WebApp for Mobile Control of NINA
- [NINA Plugins](https://nighttime-imaging.eu/plugins/) - Official NINA plugin repository

## 📞 Support

Need help? Check out:
- [Documentation](docs/)
- [Discussions](../../discussions)
- [Issues](../../issues)
