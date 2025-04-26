#!/usr/bin/env python3
"""
NINA Advanced API Server - Provides tools to Claude for controlling N.I.N.A. using the Advanced API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first, before any other imports
load_dotenv()

# Default environment variable names
ENV_NINA_HOST = 'NINA_HOST'
ENV_NINA_PORT = 'NINA_PORT'
ENV_LOG_LEVEL = 'LOG_LEVEL'
ENV_IMAGE_SAVE_DIR = 'IMAGE_SAVE_DIR'

# Default values
DEFAULT_NINA_HOST = 'localhost'
DEFAULT_NINA_PORT = 1888
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_IMAGE_SAVE_DIR = str(Path('~/Desktop/NINA_Images').expanduser())

# Get values from environment with defaults
NINA_HOST = os.getenv(ENV_NINA_HOST, DEFAULT_NINA_HOST)
NINA_PORT = int(os.getenv(ENV_NINA_PORT, DEFAULT_NINA_PORT))
LOG_LEVEL = os.getenv(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL).upper()
IMAGE_SAVE_DIR = str(Path(os.getenv(ENV_IMAGE_SAVE_DIR, DEFAULT_IMAGE_SAVE_DIR)).expanduser())

# Now import everything else
import asyncio
import logging
import sys
import json
import time
from datetime import datetime
import base64
from typing import Dict, Any, Optional, List, Union, Callable, Literal
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from enum import Enum
import aiohttp
import requests
from urllib.parse import quote

# Configure logging
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, 'nina_advanced_api.log')

# Set up logging with configured level
numeric_level = getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=numeric_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NinaAdvancedAPI')

# Add request ID to log context
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'N/A'
        return True

logger.addFilter(RequestIdFilter())

# Log initial configuration
logger.info(f"Starting with configuration:")
logger.info(f"NINA_HOST: {NINA_HOST}")
logger.info(f"NINA_PORT: {NINA_PORT}")
logger.info(f"LOG_LEVEL: {LOG_LEVEL}")
logger.info(f"IMAGE_SAVE_DIR: {IMAGE_SAVE_DIR}")

# Connection Mode enum
class ConnectionMode(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

# Server State
class ServerState:
    def __init__(self):
        self.mode = ConnectionMode.DISCONNECTED
        self.host = NINA_HOST
        self.port = NINA_PORT
        self.last_error = None
        self.session = None
        self.image_save_path = IMAGE_SAVE_DIR
        logger.info(f"Server state initialized with host: {self.host}, port: {self.port}, image save path: {self.image_save_path}")

    def to_dict(self):
        """Convert state to a dictionary for API responses"""
        return {
            "mode": self.mode,
            "host": self.host,
            "port": self.port,
            "last_error": self.last_error,
            "image_save_path": self.image_save_path
        }

    def set_error(self, error_msg):
        """Set error message and log it"""
        self.last_error = error_msg
        logger.error(error_msg)

    def clear_error(self):
        """Clear error state"""
        self.last_error = None

# Initialize server state
server_state = ServerState()

# Custom exceptions
class NinaError(Exception):
    """Base exception for NINA-related errors"""
    pass

class ImageSaveError(NinaError):
    """Exception raised when there's an error saving an image"""
    pass

class ImageDataError(NinaError):
    """Exception raised when there's an error processing image data"""
    pass

class ClientNotInitializedError(NinaError):
    """Exception raised when the NINA client is not initialized"""
    pass

# Error response templates
def create_error_response(error_type: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    """Create a standardized error response"""
    return {
        "Success": False,
        "Error": message,
        "ErrorType": error_type,
        "ErrorDetails": details or {},
        "StatusCode": 500,
        "Type": "API"
    }

# Initialize FastMCP
mcp = FastMCP(
    name="NINA Advanced API",
    description="Model Context Provider for N.I.N.A. astrophotography software using Advanced API",
    version="0.1.0"
)

class ImageType(str, Enum):
    """Valid image types for camera capture"""
    RAW = "RAW"
    FITS = "FITS"
    TIFF = "TIFF"

class NinaAPIClient:
    def __init__(self):
        load_dotenv()
        self.host = os.getenv("NINA_HOST", DEFAULT_NINA_HOST)
        self.port = os.getenv("NINA_PORT", DEFAULT_NINA_PORT)
        self.session = None
        self._connected = False

    async def connect(self):
        """Establish HTTP connection"""
        if self._connected:
            return

        try:
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            self.session = aiohttp.ClientSession(timeout=timeout)
            # Test connection with base API endpoint
            async with self.session.get(f"http://{self.host}:{self.port}/v2/api") as response:
                if response.status == 200:
                    self._connected = True
                    logger.info("HTTP connection established")
                else:
                    raise ConnectionError(f"Failed to connect: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to connect to HTTP server: {str(e)}")
            await self.close()
            raise

    async def close(self):
        """Close the HTTP connection"""
        if self.session:
            await self.session.close()
            self.session = None
        self._connected = False

    async def _send_request(self, method: str, endpoint: str, data: Optional[Dict] = None, handle_image_stream: bool = False) -> Dict[str, Any]:
        """Send a request to the NINA API"""
        if not self.session:
            raise ConnectionError("HTTP connection not established")

        url = f"http://{self.host}:{self.port}/v2/api/{endpoint}"
        try:
            logger.debug(f"Sending {method} request to {url}")
            async with self.session.request(method, url, json=data) as response:
                # Check if this is an image stream response
                content_type = response.headers.get('Content-Type', '')
                if handle_image_stream and 'image/' in content_type:
                    # This is an image stream - read binary data
                    image_data = await response.read()
                    logger.debug(f"Received image data from {url} ({len(image_data)} bytes)")
                    return {
                        "Success": True,
                        "ContentType": content_type,
                        "ImageData": image_data,
                        "IsImageStream": True
                    }
                else:    
                    # Regular JSON response
                    try:
                        result = await response.json()
                        logger.debug(f"Response from {url}: {result}")
                    except json.JSONDecodeError:
                        if 'image/' in content_type:
                            # We got an image when we weren't expecting one
                            image_data = await response.read()
                            return {
                                "Success": True,
                                "ContentType": content_type,
                                "ImageData": image_data,
                                "IsImageStream": True
                            }
                        else:
                            # Some other JSON decode error
                            raise
                    
                    # Check if the response indicates an error
                    if not result.get("Success", False):
                        error_msg = result.get("Error", "Unknown error")
                        if error_msg == "Unknown error" and result.get("Response") == "":
                            # Special case: empty response with success=false usually means the operation failed
                            error_msg = "Operation failed - no response from device"
                        logger.error(f"Error in {endpoint}: {error_msg}")
                        raise NinaError(f"Error in {endpoint}: {error_msg}")
                    return result
        except aiohttp.ClientError as e:
            logger.error(f"Network error in {endpoint}: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {endpoint}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in {endpoint}: {str(e)}")
            raise

    async def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information"""
        return await self._send_request("GET", "equipment/camera/info")

    async def connect_camera(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Connect to camera"""
        endpoint = "equipment/camera/connect"
        if device_id:
            endpoint += f"?to={quote(device_id)}"
        return await self._send_request("GET", endpoint)

    async def disconnect_camera(self) -> Dict[str, Any]:
        """Disconnect camera"""
        return await self._send_request("GET", "equipment/camera/disconnect")

    async def list_camera_devices(self) -> Dict[str, Any]:
        """List available camera devices. This will also trigger a rescan of available devices.
        
        The API automatically rescans for devices when this endpoint is called.
        """
        return await self._send_request("GET", "equipment/camera/list-devices")

    async def get_capture_statistics(self) -> Dict[str, Any]:
        """Get statistics about the last captured image.
        
        Returns statistics like:
        - Stars count
        - HFR (Half Flux Radius)
        - Median
        - Mean
        - Min/Max values
        - Standard deviation
        - Median absolute deviation
        """
        return await self._send_request("GET", "equipment/camera/capture/statistics")

    async def set_binning(self, binning: str) -> Dict[str, Any]:
        """Set camera binning mode.
        
        Args:
            binning: Binning mode in format "1x1", "2x2", "3x3", "4x4" etc.
        """
        return await self._send_request("GET", f"equipment/camera/set-binning?binning={binning}")

    async def control_dew_heater(self, power: bool) -> Dict[str, Any]:
        """Control camera's dew heater.
        
        Args:
            power: True to enable, False to disable
        """
        return await self._send_request("GET", f"equipment/camera/dew-heater?power={str(power).lower()}")

    async def capture_image(self, 
                          exposure: Optional[float] = None,
                          gain: Optional[int] = None,
                          binning: Optional[str] = None,
                          filter_name: Optional[str] = None,
                          count: Optional[int] = None,
                          image_type: Optional[ImageType] = None,
                          save: Optional[bool] = True,
                          filename: Optional[str] = None,
                          solve: Optional[bool] = None,
                          solve_timeout: Optional[int] = None) -> Dict[str, Any]:
        """Capture an image with the camera.
        
        Args:
            exposure: Exposure time in seconds
            gain: Camera gain
            binning: Binning mode e.g. "1x1", "2x2"
            filter_name: Filter name
            count: Number of exposures
            image_type: Image type (RAW, FITS, or TIFF)
            save: Whether to save the image
            filename: Custom filename
            solve: Whether to plate solve
            solve_timeout: Timeout for plate solving in seconds
        """
        params = []
        if exposure is not None:
            params.append(f"exposure={exposure}")
        if gain is not None:
            params.append(f"gain={gain}")
        if binning is not None:
            params.append(f"binning={binning}")
        if filter_name is not None:
            params.append(f"filter={filter_name}")
        if count is not None:
            params.append(f"count={count}")
        if image_type is not None:
            params.append(f"type={image_type.value}")
        if save is not None:
            params.append(f"save={str(save).lower()}")
        if filename is not None:
            params.append(f"filename={filename}")
        if solve is not None:
            params.append(f"solve={str(solve).lower()}")
        if solve_timeout is not None:
            params.append(f"solve_timeout={solve_timeout}")
            
        endpoint = "equipment/camera/capture"
        if params:
            endpoint += "?" + "&".join(params)
            
        return await self._send_request("GET", endpoint)

# Input Models
class ConnectInput(BaseModel):
    """Input model for version endpoint"""
    host: str = Field(default_factory=lambda: os.getenv(ENV_NINA_HOST, DEFAULT_NINA_HOST))
    port: int = Field(default_factory=lambda: int(os.getenv(ENV_NINA_PORT, DEFAULT_NINA_PORT)))

class FilterWheelConnectInput(BaseModel):
    device_id: Optional[str] = None

class FilterChangeInput(BaseModel):
    filter_id: int

class FilterInfoInput(BaseModel):
    filter_id: int

# Camera Input Models
class CameraConnectInput(BaseModel):
    device_id: Optional[str] = None

class CameraReadoutModeInput(BaseModel):
    mode: int

class CameraCoolingInput(BaseModel):
    """Input model for camera cooling settings"""
    temperature: float  # Target temperature in Celsius
    duration: Optional[int] = None  # Duration in minutes (not seconds)

class CameraDewHeaterInput(BaseModel):
    """Input model for camera dew heater settings"""
    power: bool  # True to enable, False to disable

class CameraBinningInput(BaseModel):
    """Input model for camera binning settings"""
    binning: str  # Format: "1x1", "2x2", "3x3", "4x4" etc.

class ImageHistoryInput(BaseModel):
    """Input model for getting image history"""
    limit: Optional[int] = None  # Limit to number of images to return
    offset: Optional[int] = None  # Offset for pagination
    all: bool = True  # Whether to get all images or only the current image
    imageType: Optional[str] = None  # Filter by image type (LIGHT, FLAT, DARK, BIAS, SNAPSHOT)
    count: Optional[bool] = None  # Whether to count the number of images

class CameraCaptureInput(BaseModel):
    """Input model for camera capture settings.
    
    This model provides a simplified interface for capturing images with NINA.
    The capture can be done in two modes:
    1. Download mode (download=True):
       - Returns the image data immediately
       - Can be resized and compressed for preview purposes
       - Supports JPEG (quality 1-100) or PNG (quality=-1) format
       - Image is still saved by NINA in full resolution
       
    2. Background capture mode (download=False):
       - Starts the capture and returns immediately
       - Does not wait for completion or return image data
       - Image is saved by NINA in full resolution
       - Useful for long exposures or when image data isn't needed
    
    Args:
        duration: The duration of the exposure in seconds
        gain: Camera gain setting (camera specific)
        download: Whether to wait for and return the image data
        resize: When downloading, whether to resize the image
        quality: When downloading, image quality (1-100 for JPEG, -1 for PNG)
        size: When downloading and resizing, target size (e.g. "1920x1080")
        solve: Whether to plate solve the image
    """
    duration: Optional[float] = None  # The duration of the exposure in seconds
    gain: Optional[int] = None  # Camera gain setting
    download: bool = False  # Whether to wait for and return the image data
    resize: Optional[bool] = None  # When downloading, whether to resize the image
    quality: Optional[int] = None  # When downloading, image quality (1-100 for JPEG, -1 for PNG)
    size: Optional[str] = None  # When downloading and resizing, target size (e.g. "1920x1080")
    solve: Optional[bool] = None  # Whether to plate solve the image

# Mount Input Models
class MountConnectInput(BaseModel):
    device_id: Optional[str] = None

class TrackingMode(str, Enum):
    """Valid tracking modes for mount"""
    SIDEREAL = "0"  # Sidereal tracking
    LUNAR = "1"     # Lunar tracking
    SOLAR = "2"     # Solar tracking
    KING = "3"      # King rate tracking
    STOPPED = "4"   # Tracking stopped

class MountTrackingModeInput(BaseModel):
    """Input model for mount tracking mode settings"""
    mode: TrackingMode  # The tracking mode to set

class MountSlewInput(BaseModel):
    """Input model for mount slew settings"""
    ra: float  # Right Ascension in hours (will be converted to degrees)
    dec: float  # Declination in degrees
    wait_for_completion: Optional[bool] = True  # Whether to wait for the slew to complete

class MountParkPositionInput(BaseModel):
    ra: float
    dec: float

# Dome Input Models
class DomeConnectInput(BaseModel):
    device_id: Optional[str] = None

class DomeSlewInput(BaseModel):
    azimuth: float

class DomeParkPositionInput(BaseModel):
    azimuth: float

class DomeFollowInput(BaseModel):
    enabled: bool

# Flats Input Models
class FlatsInput(BaseModel):
    """Input model for flats capture settings"""
    count: Optional[int] = None  # Number of flats to capture
    minExposure: Optional[float] = None  # Minimum exposure time in seconds
    maxExposure: Optional[float] = None  # Maximum exposure time in seconds
    histogramMean: Optional[float] = None  # Target histogram mean value
    meanTolerance: Optional[float] = None  # Tolerance for histogram mean
    dither: Optional[bool] = None  # Whether to dither between exposures
    filterId: Optional[int] = None  # ID of the filter to use
    binning: Optional[str] = None  # Binning mode e.g. "1x1", "2x2"
    gain: Optional[int] = None  # Camera gain setting
    offset: Optional[int] = None  # Camera offset setting

# Global client instance
nina_client = None

async def get_client() -> NinaAPIClient:
    """Get or create an HTTP client instance"""
    global nina_client
    if nina_client is None:
        nina_client = NinaAPIClient()
    return nina_client

# Filter Wheel Tools
@mcp.tool()
async def nina_connect_filterwheel(input: FilterWheelConnectInput) -> Dict[str, Any]:
    """Connect to a filter wheel device in NINA astronomy software.
    
    Args:
        input: FilterWheelConnectInput containing:
            device_id: Optional device ID to connect to. If not provided, will use default device.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the connection
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFilterWheelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Build the endpoint URL
        endpoint = "equipment/filterwheel/connect"
        if input.device_id:
            endpoint += f"?to={input.device_id}"

        # Send the connect command
        result = await client._send_request("GET", endpoint)
        
        return {
            "Success": True,
            "Message": "Filter wheel connected successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAFilterWheelError", str(e))

@mcp.tool()
async def nina_disconnect_filterwheel() -> Dict[str, Any]:
    """Disconnect the filter wheel from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the disconnection
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFilterWheelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Send the disconnect command
        result = await client._send_request("GET", "equipment/filterwheel/disconnect")
        
        return {
            "Success": True,
            "Message": "Filter wheel disconnected successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAFilterWheelError", str(e))

@mcp.tool()
async def nina_list_filterwheel_devices() -> Dict[str, Any]:
    """List available filter wheel devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available filter wheel devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFilterWheelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Get list of available filter wheel devices
        result = await client._send_request("GET", "equipment/filterwheel/list-devices")
        
        return {
            "Success": True,
            "Message": "Filter wheel devices listed successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAFilterWheelError", str(e))

@mcp.tool()
async def nina_rescan_filterwheel_devices() -> Dict[str, Any]:
    """Rescan for filter wheel devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available filter wheel devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFilterWheelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Rescan for filter wheel devices
        result = await client._send_request("GET", "equipment/filterwheel/rescan")
        
        return {
            "Success": True,
            "Message": "Filter wheel devices rescanned successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAFilterWheelError", str(e))

@mcp.tool()
async def nina_get_filterwheel_info() -> Dict[str, Any]:
    """Get filter wheel information from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Filter wheel information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFilterWheelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Get filter wheel information
        result = await client._send_request("GET", "equipment/filterwheel/info")
        
        return {
            "Success": True,
            "Message": "Filter wheel information retrieved successfully",
            "Info": result.get("Response", {}),
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAFilterWheelError", str(e))

@mcp.tool()
async def nina_change_filter(input: FilterChangeInput) -> Dict[str, Any]:
    """Change to a specific filter using the NINA astronomy software filter wheel.
    
    Args:
        input: FilterChangeInput containing:
            filter_id: ID of the filter to change to
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFilterWheelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Change to the specified filter
        result = await client._send_request("GET", f"equipment/filterwheel/change-filter?filterId={input.filter_id}")
        
        return {
            "Success": True,
            "Message": f"Filter changed to ID {input.filter_id} successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAFilterWheelError", str(e))

@mcp.tool()
async def nina_get_filter_info(input: FilterInfoInput) -> Dict[str, Any]:
    """Get information about a specific filter in the NINA astronomy software filter wheel.
    
    Args:
        input: FilterInfoInput containing:
            filter_id: ID of the filter to get information about
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Filter information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFilterWheelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Get information about the specified filter
        result = await client._send_request("GET", f"equipment/filterwheel/filter-info?filterId={input.filter_id}")
        
        return {
            "Success": True,
            "Message": f"Filter information for ID {input.filter_id} retrieved successfully",
            "Info": result.get("Response", {}),
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAFilterWheelError", str(e))

# Camera Tools
@mcp.tool()
async def nina_connect_camera(input: CameraConnectInput) -> Dict[str, Any]:
    """Connect to a camera device in NINA astronomy software.
    
    Args:
        input: CameraConnectInput containing:
            device_id: Optional device ID to connect to. If not provided, will use default device.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the connection
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        logger.info(f"Attempting to connect to camera: {input.device_id if input.device_id else 'default device'}")
        result = await client.connect_camera(input.device_id)
        
        # Check if we got a successful response
        if not result.get("Success", False):
            error_msg = result.get("Error", "Unknown error")
            logger.error(f"Failed to connect to camera: {error_msg}")
            return create_error_response(
                "NINACameraError",
                f"Failed to connect to camera: {error_msg}",
                {"StatusCode": result.get("StatusCode", 500)}
            )
            
        logger.info("Camera connected successfully")
        return {
            "Success": True,
            "Message": "Camera connected successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except NinaError as e:
        logger.error(f"NINA API error while connecting camera: {str(e)}")
        return create_error_response("NINACameraError", str(e))
    except Exception as e:
        logger.error(f"Unexpected error while connecting camera: {str(e)}")
        return create_error_response("NINACameraError", f"Unexpected error: {str(e)}")

@mcp.tool()
async def nina_disconnect_camera() -> Dict[str, Any]:
    """Disconnect the camera from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the disconnection
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client.disconnect_camera()
        return {
            "Success": True,
            "Message": "Camera disconnected successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINACameraError", str(e))

@mcp.tool()
async def nina_list_camera_devices() -> Dict[str, Any]:
    """List available camera devices in NINA astronomy software.
    This will also trigger a rescan of available devices.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available camera devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client.list_camera_devices()
        return {
            "Success": True,
            "Message": "Camera devices listed successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINACameraError", str(e))

@mcp.tool()
async def nina_get_camera_info() -> Dict[str, Any]:
    """Get camera information from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Camera information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client.get_camera_info()
        return {
            "Success": True,
            "Message": "Camera information retrieved successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINACameraError", str(e))

@mcp.tool()
async def nina_set_readout_mode(input: CameraReadoutModeInput) -> Dict[str, Any]:
    """Set the readout mode for the camera in NINA astronomy software.
    
    Args:
        input: CameraReadoutModeInput containing:
            mode: The readout mode to set
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Set the readout mode
        result = await client._send_request("GET", f"equipment/camera/set-readout-mode?mode={input.mode}")
        
        return {
            "Success": True,
            "Message": f"Readout mode set to {input.mode} successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINACameraError", str(e))

@mcp.tool()
async def nina_start_cooling(input: CameraCoolingInput) -> Dict[str, Any]:
    """Start cooling the camera in NINA astronomy software.
    
    Args:
        input: CameraCoolingInput containing:
            temperature: Target temperature in Celsius
            duration: Optional duration in minutes (not seconds)
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Build the endpoint URL with correct parameters
        endpoint = f"equipment/camera/cool?temperature={input.temperature}"
        if input.duration is not None:
            endpoint += f"&minutes={input.duration}"
        endpoint += "&cancel=false"  # Explicitly set cancel to false

        # Start cooling
        result = await client._send_request("GET", endpoint)
        
        if not result.get("Success", False):
            return create_error_response(
                "NINACameraError",
                result.get("Error", "Unknown error starting camera cooling"),
                {"StatusCode": result.get("StatusCode", 500)}
            )
        
        return {
            "Success": True,
            "Message": f"Camera cooling started to {input.temperature}°C successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINACameraError", str(e))

@mcp.tool()
async def nina_stop_cooling() -> Dict[str, Any]:
    """Stop the camera's cooling process in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Send the stop cooling command
        result = await client._send_request("GET", "equipment/camera/cool?cancel=true")
        
        if not result.get("Success", False):
            return create_error_response(
                "NINACameraError",
                result.get("Error", "Unknown error stopping camera cooling"),
                {"StatusCode": result.get("StatusCode", 500)}
            )
        
        return {
            "Success": True,
            "Message": "Camera cooling stopped successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINACameraError", str(e))

@mcp.tool()
async def nina_abort_exposure() -> Dict[str, Any]:
    """Abort the current camera exposure in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Abort exposure
        result = await client._send_request("GET", "equipment/camera/abort-exposure")
        
        return {
            "Success": True,
            "Message": "Exposure aborted successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINACameraError", str(e))

@mcp.tool()
async def nina_control_dew_heater(input: CameraDewHeaterInput) -> Dict[str, Any]:
    """Control the camera's dew heater in NINA astronomy software.
    
    Args:
        input: CameraDewHeaterInput containing:
            power: True to enable, False to disable
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
        
    Note: Not all cameras have dew heater capability. The operation will fail
    with an appropriate error message if the camera doesn't support it.
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client.control_dew_heater(input.power)
        
        # Check for specific error about dew heater not being available
        if not result.get("Success", False) and "has no dew heater" in result.get("Error", ""):
            return create_error_response(
                "NINACameraError",
                "Camera does not have dew heater capability",
                {"StatusCode": 409}
            )
            
        return {
            "Success": True,
            "Message": f"Dew heater {'enabled' if input.power else 'disabled'} successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINACameraError", str(e))

@mcp.tool()
async def nina_set_binning(input: CameraBinningInput) -> Dict[str, Any]:
    """Set the camera's binning mode in NINA astronomy software.
    
    Args:
        input: CameraBinningInput containing:
            binning: Binning mode in format "1x1", "2x2", "3x3", "4x4" etc.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client.set_binning(input.binning)
        return {
            "Success": True,
            "Message": f"Binning set to {input.binning} successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINACameraError", str(e))

@mcp.tool()
async def nina_capture_image(input: CameraCaptureInput) -> Dict[str, Any]:
    """Capture an image with NINA astronomy software.
    
    This function provides two modes of operation:
    1. Download mode (input.download=True):
       - Waits for the capture to complete
       - Saves the image to disk and returns the file path
       - Can resize and compress the image for preview
       - Supports JPEG (quality 1-100) or PNG (quality=-1) format
       - Image is also saved by NINA in full resolution
       
    2. Background capture mode (input.download=False):
       - Starts the capture and returns immediately
       - Does not wait for completion
       - Image is saved by NINA in full resolution
       - Useful for long exposures
    """
    try:
        client = await get_client()
        if not client._connected:
            await client.connect()

        # Calculate dynamic timeout based on operation
        base_timeout = 10  # Base timeout in seconds
        total_timeout = base_timeout
        
        if input.duration is not None:
            # Add exposure time plus buffer for capture
            total_timeout += input.duration + 10
            
        if input.solve:
            # Add platesolve timeout
            total_timeout += 120
            
        # Create new client session with dynamic timeout
        if client.session:
            await client.session.close()
        timeout = aiohttp.ClientTimeout(total=total_timeout)
        client.session = aiohttp.ClientSession(timeout=timeout)
        logger.info(f"Set dynamic timeout to {total_timeout} seconds for capture operation")

        # Build query parameters
        params = []
        
        # Required parameters
        params.append("save=true")  # Always save in NINA
        
        if input.duration is not None:
            params.append(f"duration={input.duration}")
        if input.gain is not None:
            params.append(f"gain={input.gain}")
        if input.solve is not None:
            params.append(f"solve={str(input.solve).lower()}")
            
        # Handle download mode settings
        if input.download:
            params.append("stream=true")  # Use streaming for downloads
            params.append("waitForResult=true")  # Wait for capture to complete
            
            if input.resize:
                params.append("resize=true")
                if input.size:
                    params.append(f"size={input.size}")
            if input.quality is not None:
                params.append(f"quality={input.quality}")
        else:
            params.append("omitImage=true")  # Don't return image data
            params.append("waitForResult=false")  # Don't wait for completion

        # Build the endpoint
        endpoint = "equipment/camera/capture"
        if params:
            endpoint += "?" + "&".join(params)

        logger.info(f"Sending capture request to endpoint: {endpoint}")

        try:
            result = await client._send_request("GET", endpoint, handle_image_stream=input.download)
            
            # For successful capture
            if result.get("Success", False):
                response = {
                    "Success": True,
                    "Message": "Image captured successfully",
                    "Type": "NINA_API"
                }
            
                # Handle image data if present
                if result.get("IsImageStream", False):
                    try:
                        # Ensure save directory exists
                        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
                        
                        # Generate filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        ext = "png" if input.quality == -1 else "jpg"
                        filename = f"nina_capture_{timestamp}.{ext}"
                        file_path = os.path.join(IMAGE_SAVE_DIR, filename)
                        
                        # Save the image
                        with open(file_path, 'wb') as f:
                            f.write(result["ImageData"])
                            
                        response["SavedPath"] = file_path
                        logger.info(f"Image saved to: {file_path}")
                    except Exception as e:
                        logger.error(f"Error saving image: {str(e)}")
                        return create_error_response(
                            "NINACameraError",
                            f"Failed to save image: {str(e)}",
                            {"StatusCode": 500}
                        )
                
                # Handle plate solve results if present
                if "PlateSolveResult" in result.get("Response", {}):
                    solve_result = result["Response"]["PlateSolveResult"]
                    response["PlateSolveResult"] = {
                        "Success": solve_result.get("Success", False),
                        "RA": solve_result.get("RA"),
                        "Dec": solve_result.get("Dec"),
                        "Rotation": solve_result.get("Rotation"),
                        "PixelScale": solve_result.get("PixelScale"),
                        "Error": solve_result.get("Error")
                    }
            
                # Include any other details from the response
                if "Response" in result:
                    response["Details"] = result["Response"]
                
                return response
            else:
                # Return the error from the API
                return create_error_response(
                    "NINACameraError",
                    result.get("Error", "Unknown error during capture"),
                    {"StatusCode": result.get("StatusCode", 500)}
                )
                
        except asyncio.TimeoutError:
            logger.error(f"Capture operation timed out after {total_timeout} seconds")
            # Try to abort the exposure if we timeout
            try:
                await client._send_request("GET", "equipment/camera/abort-exposure")
            except Exception as abort_error:
                logger.error(f"Failed to abort exposure after timeout: {str(abort_error)}")
            return create_error_response(
                "NINACameraError",
                f"Capture operation timed out after {total_timeout} seconds",
                {"StatusCode": 408}
            )
        except Exception as e:
            logger.error(f"Error processing capture request: {str(e)}")
            return create_error_response(
                "NINACameraError",
                f"Error processing capture request: {str(e)}",
                {"StatusCode": 500}
            )
            
    except Exception as e:
        logger.error(f"Error in nina_capture_image: {str(e)}")
        return create_error_response(
            "NINACameraError",
            str(e),
            {"StatusCode": 500}
        )

@mcp.tool()
async def nina_get_capture_statistics() -> Dict[str, Any]:
    """Get statistics about the last captured image in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Statistics including:
            - Stars count
            - HFR (Half Flux Radius)
            - Median
            - Mean
            - Min/Max values
            - Standard deviation
            - Median absolute deviation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client.get_capture_statistics()
        return {
            "Success": True,
            "Message": "Capture statistics retrieved successfully",
            "Statistics": result.get("Response", {}),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINACameraError", str(e))

@mcp.tool()
async def nina_get_image_history(input: ImageHistoryInput) -> Dict[str, Any]:
    """Get image history from NINA astronomy software.
    
    Args:
        input: ImageHistoryInput containing:
            limit: Optional limit to number of images to return
            offset: Optional offset for pagination
            all: Whether to get all images or only the current image (defaults to True)
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Image history
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAImageError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Build query parameters
        params = []
        params.append(f"all={str(input.all).lower()}")
        
        if input.limit is not None:
            params.append(f"limit={input.limit}")
        if input.offset is not None:
            params.append(f"offset={input.offset}")
        if input.imageType is not None:
            params.append(f"imageType={input.imageType}")
        if input.count is not None:
            params.append(f"count={str(input.count).lower()}")

        endpoint = "image-history"
        if params:
            endpoint += "?" + "&".join(params)

        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": "Image history retrieved successfully",
            "Details": result.get("Response", {}),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAImageError", str(e))

# Mount Tools
@mcp.tool()
async def nina_connect_mount(input: MountConnectInput) -> Dict[str, Any]:
    """Connect to a mount device in NINA astronomy software.
    
    Args:
        input: MountConnectInput containing:
            device_id: Optional device ID to connect to. If not provided, will use default device.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the connection
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Build the endpoint URL
        endpoint = "equipment/mount/connect"
        if input.device_id:
            endpoint += f"?to={input.device_id}"

        # Send the connect command
        result = await client._send_request("GET", endpoint)
        
        return {
            "Success": True,
            "Message": "Mount connected successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_disconnect_mount() -> Dict[str, Any]:
    """Disconnect the mount from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the disconnection
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Send the disconnect command
        result = await client._send_request("GET", "equipment/mount/disconnect")
        
        return {
            "Success": True,
            "Message": "Mount disconnected successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_list_mount_devices() -> Dict[str, Any]:
    """List available mount devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available mount devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Get list of available mount devices
        result = await client._send_request("GET", "equipment/mount/list-devices")
        
        return {
            "Success": True,
            "Message": "Mount devices listed successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_rescan_mount_devices() -> Dict[str, Any]:
    """Rescan for mount devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available mount devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Rescan for mount devices
        result = await client._send_request("GET", "equipment/mount/rescan")
        
        return {
            "Success": True,
            "Message": "Mount devices rescanned successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_get_mount_info() -> Dict[str, Any]:
    """Get mount information from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Mount information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Get mount information
        result = await client._send_request("GET", "equipment/mount/info")
        
        return {
            "Success": True,
            "Message": "Mount information retrieved successfully",
            "Info": result.get("Response", {}),
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_home_mount() -> Dict[str, Any]:
    """Send the mount to its home position in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Send mount to home position
        result = await client._send_request("GET", "equipment/mount/home")
        
        return {
            "Success": True,
            "Message": "Mount sent to home position successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_set_tracking_mode(input: MountTrackingModeInput) -> Dict[str, Any]:
    """Set the mount's tracking mode in NINA astronomy software.
    
    Args:
        input: MountTrackingModeInput containing:
            mode: The tracking mode to set:
                - SIDEREAL (0): Sidereal tracking
                - LUNAR (1): Lunar tracking
                - SOLAR (2): Solar tracking
                - KING (3): King rate tracking
                - STOPPED (4): Tracking stopped
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Set tracking mode using the numeric value
        result = await client._send_request("GET", f"equipment/mount/tracking?mode={input.mode.value}")
        
        if not result.get("Success", False):
            return create_error_response(
                "NINAMountError",
                result.get("Error", "Unknown error setting tracking mode"),
                {"StatusCode": result.get("StatusCode", 500)}
            )
        
        return {
            "Success": True,
            "Message": f"Tracking mode set to {input.mode.name} successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_park_mount() -> Dict[str, Any]:
    """Park the mount in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Park the mount
        result = await client._send_request("GET", "equipment/mount/park")
        
        return {
            "Success": True,
            "Message": "Mount parked successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_unpark_mount() -> Dict[str, Any]:
    """Unpark the mount in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Unpark the mount
        result = await client._send_request("GET", "equipment/mount/unpark")
        
        return {
            "Success": True,
            "Message": "Mount unparked successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_flip_mount() -> Dict[str, Any]:
    """Flip the mount in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Flip the mount
        result = await client._send_request("GET", "equipment/mount/flip")
        
        return {
            "Success": True,
            "Message": "Mount flipped successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_slew_mount(input: MountSlewInput) -> Dict[str, Any]:
    """Slew the mount to specified coordinates in NINA astronomy software.
    
    Args:
        input: MountSlewInput containing:
            ra: Right Ascension in hours (0-24)
            dec: Declination in degrees (-90 to +90)
            wait_for_completion: Whether to wait for the slew to complete
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Validate coordinates
        if not 0 <= input.ra <= 24:
            return create_error_response(
                "NINAMountError", 
                "Right Ascension must be between 0 and 24 hours",
                {"StatusCode": 400}
            )
        if not -90 <= input.dec <= 90:
            return create_error_response(
                "NINAMountError",
                "Declination must be between -90 and +90 degrees",
                {"StatusCode": 400}
            )

        # Convert RA from hours to degrees (1 hour = 15 degrees)
        ra_degrees = input.ra * 15.0

        # Build the endpoint URL with correct parameters
        endpoint = f"equipment/mount/slew?ra={ra_degrees}&dec={input.dec}&waitForResult={str(input.wait_for_completion).lower()}"

        # Send the slew command
        result = await client._send_request("GET", endpoint)
        
        if not result.get("Success", False):
            return create_error_response(
                "NINAMountError",
                result.get("Error", "Unknown error during slew"),
                {"StatusCode": result.get("StatusCode", 500)}
            )

        # Get appropriate message based on wait setting
        if input.wait_for_completion:
            message = f"Mount slewed to RA={input.ra}h, Dec={input.dec}° successfully"
        else:
            message = f"Mount slew to RA={input.ra}h, Dec={input.dec}° started"
        
        return {
            "Success": True,
            "Message": message,
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_stop_slew() -> Dict[str, Any]:
    """Stop the mount's current slew in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Stop the slew
        result = await client._send_request("GET", "equipment/mount/stop-slew")
        
        return {
            "Success": True,
            "Message": "Mount slew stopped successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

@mcp.tool()
async def nina_set_park_position(input: MountParkPositionInput) -> Dict[str, Any]:
    """Set the mount's park position in NINA astronomy software.
    
    Args:
        input: MountParkPositionInput containing:
            ra: Right Ascension in hours
            dec: Declination in degrees
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAMountError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Set park position
        result = await client._send_request("GET", f"equipment/mount/set-park-position?ra={input.ra}&dec={input.dec}")
        
        return {
            "Success": True,
            "Message": f"Park position set to RA={input.ra}h, Dec={input.dec}° successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAMountError", str(e))

# Dome Tools
@mcp.tool()
async def nina_connect_dome(input: DomeConnectInput) -> Dict[str, Any]:
    """Connect to a dome device in NINA astronomy software.
    
    Args:
        input: DomeConnectInput containing:
            device_id: Optional device ID to connect to. If not provided, will use default device.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the connection
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Build the endpoint URL
        endpoint = "equipment/dome/connect"
        if input.device_id:
            endpoint += f"?to={input.device_id}"

        # Send the connect command
        result = await client._send_request("GET", endpoint)
        
        return {
            "Success": True,
            "Message": "Dome connected successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_disconnect_dome() -> Dict[str, Any]:
    """Disconnect the dome from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the disconnection
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Send the disconnect command
        result = await client._send_request("GET", "equipment/dome/disconnect")
        
        return {
            "Success": True,
            "Message": "Dome disconnected successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_list_dome_devices() -> Dict[str, Any]:
    """List available dome devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available dome devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Get list of available dome devices
        result = await client._send_request("GET", "equipment/dome/list-devices")
        
        return {
            "Success": True,
            "Message": "Dome devices listed successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_rescan_dome_devices() -> Dict[str, Any]:
    """Rescan for dome devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available dome devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Rescan for dome devices
        result = await client._send_request("GET", "equipment/dome/rescan")
        
        return {
            "Success": True,
            "Message": "Dome devices rescanned successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_get_dome_info() -> Dict[str, Any]:
    """Get dome information from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Dome information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Get dome information
        result = await client._send_request("GET", "equipment/dome/info")
        
        return {
            "Success": True,
            "Message": "Dome information retrieved successfully",
            "Info": result.get("Response", {}),
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_open_dome_shutter() -> Dict[str, Any]:
    """Open the dome shutter in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Open the shutter
        result = await client._send_request("GET", "equipment/dome/open-shutter")
        
        return {
            "Success": True,
            "Message": "Dome shutter opened successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_close_dome_shutter() -> Dict[str, Any]:
    """Close the dome shutter in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Close the shutter
        result = await client._send_request("GET", "equipment/dome/close-shutter")
        
        return {
            "Success": True,
            "Message": "Dome shutter closed successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_stop_dome_movement() -> Dict[str, Any]:
    """Stop the dome's current movement in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Stop dome movement
        result = await client._send_request("GET", "equipment/dome/stop-movement")
        
        return {
            "Success": True,
            "Message": "Dome movement stopped successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_set_dome_follow(input: DomeFollowInput) -> Dict[str, Any]:
    """Set whether the dome follows the telescope in NINA astronomy software.
    
    Args:
        input: DomeFollowInput containing:
            enabled: Whether to enable dome following
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Set dome follow
        result = await client._send_request("GET", f"equipment/dome/set-follow?enabled={str(input.enabled).lower()}")
        
        return {
            "Success": True,
            "Message": f"Dome follow {'enabled' if input.enabled else 'disabled'} successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_sync_dome_to_telescope() -> Dict[str, Any]:
    """Sync the dome position to the telescope position in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Sync dome to telescope
        result = await client._send_request("GET", "equipment/dome/sync-to-telescope")
        
        return {
            "Success": True,
            "Message": "Dome synced to telescope successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_slew_dome(input: DomeSlewInput) -> Dict[str, Any]:
    """Slew the dome to specified azimuth in NINA astronomy software.
    
    Args:
        input: DomeSlewInput containing:
            azimuth: Target azimuth in degrees
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Slew the dome
        result = await client._send_request("GET", f"equipment/dome/slew?azimuth={input.azimuth}")
        
        return {
            "Success": True,
            "Message": f"Dome slewed to {input.azimuth}° successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_set_dome_park_position(input: DomeParkPositionInput) -> Dict[str, Any]:
    """Set the dome's park position in NINA astronomy software.
    
    Args:
        input: DomeParkPositionInput containing:
            azimuth: Park position azimuth in degrees
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Set park position
        result = await client._send_request("GET", f"equipment/dome/set-park-position?azimuth={input.azimuth}")
        
        return {
            "Success": True,
            "Message": f"Dome park position set to {input.azimuth}° successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_park_dome() -> Dict[str, Any]:
    """Park the dome in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Park the dome
        result = await client._send_request("GET", "equipment/dome/park")
        
        return {
            "Success": True,
            "Message": "Dome parked successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_home_dome() -> Dict[str, Any]:
    """Send the dome to its home position in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINADomeError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Send dome to home position
        result = await client._send_request("GET", "equipment/dome/home")
        
        return {
            "Success": True,
            "Message": "Dome sent to home position successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINADomeError", str(e))

@mcp.tool()
async def nina_wait(seconds: float) -> Dict[str, Any]:
    """Wait for a specified duration in seconds.
    
    Args:
        seconds: Duration to wait in seconds (can be fractional)
        
    Returns:
        Dict containing:
        - Success status
        - Message about the wait operation
        - Type of response
    """
    try:
        if seconds < 0:
            return create_error_response(
                "InvalidParameter",
                "Wait duration cannot be negative",
                {"StatusCode": 400}
            )
            
        logger.info(f"Waiting for {seconds} seconds...")
        await asyncio.sleep(seconds)
        
        return {
            "Success": True,
            "Message": f"Waited for {seconds} seconds",
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response(
            "WaitError",
            str(e),
            {"StatusCode": 500}
        )

@mcp.tool()
async def nina_connect(input: ConnectInput) -> Dict[str, Any]:
    """Connect to the NINA astronomy software HTTP server. This tool is specifically for NINA control and should not be used for direct hardware connections."""
    try:
        # Always use environment variables if available
        server_state.host = os.getenv(ENV_NINA_HOST, input.host)
        server_state.port = int(os.getenv(ENV_NINA_PORT, input.port))
        
        client = await get_client()
        client.host = server_state.host
        client.port = server_state.port
        await client.connect()
        
        server_state.mode = ConnectionMode.CONNECTED
        server_state.clear_error()
        return {
            "Success": True,
            "Message": f"Connected to N.I.N.A. at {server_state.host}:{server_state.port}",
            "State": server_state.to_dict(),
            "Type": "NINA_API"
        }
    except Exception as e:
        server_state.set_error(str(e))
        return create_error_response("NINAConnectionError", str(e))

@mcp.tool()
async def nina_disconnect() -> Dict[str, Any]:
    """Disconnect from the NINA astronomy software HTTP server. This tool is specifically for NINA control and should not be used for direct hardware connections."""
    try:
        client = await get_client()
        await client.close()
        server_state.mode = ConnectionMode.DISCONNECTED
        server_state.clear_error()
        return {
            "Success": True,
            "Message": "Disconnected from NINA HTTP server",
            "Type": "NINA_API"
        }
    except Exception as e:
        server_state.set_error(str(e))
        return create_error_response("NINADisconnectionError", str(e))

@mcp.tool()
async def nina_time_now() -> Dict[str, Any]:
    """Get the current time from the computer in various formats.
    
    Returns:
        Dict containing:
        - local_time: Current local time in ISO format
        - utc_time: Current UTC time in ISO format
        - timestamp: Unix timestamp
        - formatted_time: Formatted time string (YYYY-MM-DD HH:MM:SS)
    """
    try:
        now = datetime.now()
        utc_now = datetime.utcnow()
        
        return {
            "Success": True,
            "Message": "Current time retrieved successfully",
            "Details": {
                "local_time": now.isoformat(),
                "utc_time": utc_now.isoformat(),
                "timestamp": time.time(),
                "formatted_time": now.strftime("%Y-%m-%d %H:%M:%S")
            },
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response(
            "TimeError",
            str(e),
            {"StatusCode": 500}
        )

@mcp.tool()
async def nina_get_status() -> Dict[str, Any]:
    """Get the current status of all connected equipment.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Status information for all equipment including:
            - Camera
            - Mount
            - Focuser
            - Filter Wheel
            - Guider
            - Dome
            - Flat Panel
            - Safety Monitor
            - Weather
            - Switches
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAStatusError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # First check if server is responsive
        try:
            health_check = await client.session.get(
                f"http://{client.host}:{client.port}/v2/api/version"
            )
            if health_check.status != 200:
                return create_error_response(
                    "NINAStatusError",
                    f"Server returned status {health_check.status}. Please ensure NINA is running and the API is enabled.",
                    {"StatusCode": health_check.status}
                )
        except Exception as e:
            return create_error_response(
                "NINAStatusError",
                f"Failed to connect to NINA server: {str(e)}",
                {"StatusCode": 500}
            )

        # Get status of each equipment type
        equipment_types = {
            "camera": "equipment/camera/info",
            "mount": "equipment/mount/info",
            "focuser": "equipment/focuser/info",
            "filterwheel": "equipment/filterwheel/info",
            "guider": "equipment/guider/info",
            "dome": "equipment/dome/info",
            "flatpanel": "equipment/flat-panel/info",
            "safetymonitor": "equipment/safetymonitor/info",
            "weather": "equipment/weather/info",
            "switch": "equipment/switch/info"
        }
        
        status = {}
        
        for eq_type, endpoint in equipment_types.items():
            try:
                result = await client._send_request("GET", endpoint)
                if result.get("Success", False):
                    status[eq_type] = {
                        "status": "ok",
                        "connected": result.get("Response", {}).get("Connected", False),
                        "info": result.get("Response", {})
                    }
                else:
                    status[eq_type] = {
                        "status": "error",
                        "message": result.get("Error", "Unknown error"),
                        "connected": False
                    }
            except Exception as e:
                logger.warning(f"Failed to get {eq_type} status: {str(e)}")
                status[eq_type] = {
                    "status": "error",
                    "message": str(e),
                    "connected": False
                }

        # Add server connection status
        status["server"] = {
            "status": "ok",
            "connected": True,
            "host": client.host,
            "port": client.port
        }

        # Count connected devices
        connected_count = sum(1 for device in status.values() 
                            if device.get("connected", False))

        return {
            "Success": True,
            "Message": f"Successfully retrieved equipment status. {connected_count} devices connected.",
            "Status": status,
            "Type": "NINA_API"
        }
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return create_error_response(
            "NINAStatusError",
            f"Failed to get status: {str(e)}",
            {"StatusCode": 500}
        )

# Flats Tools
@mcp.tool()
async def nina_sky_flats(input: FlatsInput) -> Dict[str, Any]:
    """Start capturing sky flats in NINA astronomy software.
    
    Args:
        input: FlatsInput containing:
            count: Optional number of flats to capture
            minExposure: Optional minimum exposure time in seconds
            maxExposure: Optional maximum exposure time in seconds
            histogramMean: Optional target histogram mean value
            meanTolerance: Optional tolerance for histogram mean
            dither: Optional whether to dither between exposures
            filterId: Optional ID of the filter to use
            binning: Optional binning mode e.g. "1x1", "2x2"
            gain: Optional camera gain setting
            offset: Optional camera offset setting
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatsError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Build query parameters
        params = []
        if input.count is not None:
            params.append(f"count={input.count}")
        if input.minExposure is not None:
            params.append(f"minExposure={input.minExposure}")
        if input.maxExposure is not None:
            params.append(f"maxExposure={input.maxExposure}")
        if input.histogramMean is not None:
            params.append(f"histogramMean={input.histogramMean}")
        if input.meanTolerance is not None:
            params.append(f"meanTolerance={input.meanTolerance}")
        if input.dither is not None:
            params.append(f"dither={str(input.dither).lower()}")
        if input.filterId is not None:
            params.append(f"filterId={input.filterId}")
        if input.binning:
            params.append(f"binning={input.binning}")
        if input.gain is not None:
            params.append(f"gain={input.gain}")
        if input.offset is not None:
            params.append(f"offset={input.offset}")

        endpoint = "flats/skyflat"
        if params:
            endpoint += "?" + "&".join(params)

        result = await client._send_request("GET", endpoint)
        
        return {
            "Success": True,
            "Message": "Sky flats capture started successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINAFlatsError", str(e))

@mcp.tool()
async def nina_start_flats(input: FlatsInput) -> Dict[str, Any]:
    """Start capturing flats in NINA astronomy software.
    
    Args:
        input: FlatsInput containing:
            count: Optional number of flats to capture
            minExposure: Optional minimum exposure time in seconds
            maxExposure: Optional maximum exposure time in seconds
            histogramMean: Optional target histogram mean value
            meanTolerance: Optional tolerance for histogram mean
            dither: Optional whether to dither between exposures
            filterId: Optional ID of the filter to use
            binning: Optional binning mode e.g. "1x1", "2x2"
            gain: Optional camera gain setting
            offset: Optional camera offset setting
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatsError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Build query parameters
        params = []
        if input.count is not None:
            params.append(f"count={input.count}")
        if input.minExposure is not None:
            params.append(f"minExposure={input.minExposure}")
        if input.maxExposure is not None:
            params.append(f"maxExposure={input.maxExposure}")
        if input.histogramMean is not None:
            params.append(f"histogramMean={input.histogramMean}")
        if input.meanTolerance is not None:
            params.append(f"meanTolerance={input.meanTolerance}")
        if input.dither is not None:
            params.append(f"dither={str(input.dither).lower()}")
        if input.filterId is not None:
            params.append(f"filterId={input.filterId}")
        if input.binning:
            params.append(f"binning={input.binning}")
        if input.gain is not None:
            params.append(f"gain={input.gain}")
        if input.offset is not None:
            params.append(f"offset={input.offset}")

        endpoint = "flats/start"
        if params:
            endpoint += "?" + "&".join(params)

        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": "Flats capture started successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatsError", str(e))

@mcp.tool()
async def nina_stop_flats() -> Dict[str, Any]:
    """Stop the current flats capture in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatsError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "flats/stop")
        return {
            "Success": True,
            "Message": "Flats capture stopped successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatsError", str(e))

@mcp.tool()
async def nina_get_flats_status() -> Dict[str, Any]:
    """Get the current status of flats capture in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Current status information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatsError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "flats/status")
        return {
            "Success": True,
            "Message": "Flats status retrieved successfully",
            "Status": result.get("Response", {}),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatsError", str(e))

@mcp.tool()
async def nina_get_flats_progress() -> Dict[str, Any]:
    """Get the progress of the current flats capture in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Progress information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatsError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "flats/progress")
        return {
            "Success": True,
            "Message": "Flats progress retrieved successfully",
            "Progress": result.get("Response", {}),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatsError", str(e))

# Guider Tools
@mcp.tool()
async def nina_get_guider_info() -> Dict[str, Any]:
    """Get detailed information about the connected guider.
    
    Returns information including:
    - Connection status
    - Capabilities
    - Pixel scale
    - Current state
    """
    try:
        client = await get_client()
        result = await client._send_request("GET", "equipment/guider/info")
        return {
            "Success": True,
            "Message": "Guider information retrieved successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("GuiderError", str(e))

@mcp.tool()
async def nina_connect_guider(device_id: Optional[str] = None) -> Dict[str, Any]:
    """Connect to a guider device.
    
    Args:
        device_id: Optional device ID to connect to. If not provided, will use default device.
    """
    try:
        client = await get_client()
        endpoint = "equipment/guider/connect"
        if device_id:
            endpoint += f"?to={device_id}"
        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": "Guider connected successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("GuiderError", str(e))

@mcp.tool()
async def nina_disconnect_guider() -> Dict[str, Any]:
    """Disconnect the guider."""
    try:
        client = await get_client()
        result = await client._send_request("GET", "equipment/guider/disconnect")
        return {
            "Success": True,
            "Message": "Guider disconnected successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("GuiderError", str(e))

@mcp.tool()
async def nina_list_guider_devices() -> Dict[str, Any]:
    """List all available guider devices that can be connected."""
    try:
        client = await get_client()
        result = await client._send_request("GET", "equipment/guider/list-devices")
        return {
            "Success": True,
            "Message": "Guider devices listed successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("GuiderError", str(e))

@mcp.tool()
async def nina_rescan_guider_devices() -> Dict[str, Any]:
    """Rescan for available guider devices."""
    try:
        client = await get_client()
        result = await client._send_request("GET", "equipment/guider/rescan")
        return {
            "Success": True,
            "Message": "Guider devices rescanned successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("GuiderError", str(e))

@mcp.tool()
async def nina_start_guiding(calibrate: bool = False) -> Dict[str, Any]:
    """Start guiding with optional calibration.
    
    Args:
        calibrate: If True, calibrate before starting guiding
    """
    try:
        client = await get_client()
        endpoint = "equipment/guider/start"
        if calibrate:
            endpoint += "?calibrate=true"
        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": "Guiding started successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("GuiderError", str(e))

@mcp.tool()
async def nina_stop_guiding() -> Dict[str, Any]:
    """Stop the current guiding operation."""
    try:
        client = await get_client()
        result = await client._send_request("GET", "equipment/guider/stop")
        return {
            "Success": True,
            "Message": "Guiding stopped successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("GuiderError", str(e))

@mcp.tool()
async def nina_get_guider_graph() -> Dict[str, Any]:
    """Get guider graph data for analysis."""
    try:
        client = await get_client()
        result = await client._send_request("GET", "equipment/guider/graph")
        return {
            "Success": True,
            "Message": "Guider graph data retrieved successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("GuiderError", str(e))

@mcp.tool()
async def nina_calibrate_guider() -> Dict[str, Any]:
    """Calibrate the guider without starting guiding."""
    try:
        client = await get_client()
        result = await client._send_request("GET", "equipment/guider/calibrate")
        return {
            "Success": True,
            "Message": "Guider calibration started successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("GuiderError", str(e))

@mcp.tool()
async def nina_clear_guider_calibration() -> Dict[str, Any]:
    """Clear the guider's calibration data."""
    try:
        client = await get_client()
        result = await client._send_request("GET", "equipment/guider/clear-calibration")
        return {
            "Success": True,
            "Message": "Guider calibration cleared successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("GuiderError", str(e))

class ApplicationVersionInput(BaseModel):
    """Input model for version endpoint"""
    pass

class ApplicationSwitchTabInput(BaseModel):
    """Input model for switch tab endpoint"""
    tab: str

class ApplicationPluginsInput(BaseModel):
    """Input model for plugins endpoint"""
    pass

class ApplicationScreenshotInput(BaseModel):
    """Input model for screenshot endpoint"""
    pass

@mcp.tool()
async def nina_get_version(input: ApplicationVersionInput) -> Dict[str, Any]:
    """Get the version of the NINA Advanced API.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Version information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAApplicationError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "application/version")
        return {
            "Success": True,
            "Message": "Version information retrieved successfully",
            "Version": result.get("Response", {}),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAApplicationError", str(e))

@mcp.tool()
async def nina_switch_tab(input: ApplicationSwitchTabInput) -> Dict[str, Any]:
    """Switch to a specific tab in the NINA application.
    
    Args:
        input: ApplicationSwitchTabInput containing:
            tab: Name of the tab to switch to
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAApplicationError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", f"application/switch-tab?tab={input.tab}")
        return {
            "Success": True,
            "Message": f"Switched to tab {input.tab} successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAApplicationError", str(e))

@mcp.tool()
async def nina_get_plugins(input: ApplicationPluginsInput) -> Dict[str, Any]:
    """Get information about installed plugins in NINA.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of installed plugins
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAApplicationError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "application/plugins")
        return {
            "Success": True,
            "Message": "Plugin information retrieved successfully",
            "Plugins": result.get("Response", []),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAApplicationError", str(e))

@mcp.tool()
async def nina_get_screenshot(input: ApplicationScreenshotInput) -> Dict[str, Any]:
    """Get a screenshot of the NINA application.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Base64 encoded screenshot image
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAApplicationError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "application/screenshot")
        return {
            "Success": True,
            "Message": "Screenshot retrieved successfully",
            "Screenshot": result.get("Response", {}).get("Image", ""),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAApplicationError", str(e))

class CameraWarmingInput(BaseModel):
    """Input model for camera warming settings"""
    duration: Optional[int] = None  # Duration in minutes (not seconds)
    cancel: Optional[bool] = False  # Whether to cancel the warming process

@mcp.tool()
async def nina_start_warming(input: CameraWarmingInput) -> Dict[str, Any]:
    """Start warming the camera in NINA astronomy software.
    
    Args:
        input: CameraWarmingInput containing:
            duration: Optional duration in minutes (not seconds)
            cancel: Optional whether to cancel the warming process
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINACameraError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Build the endpoint URL with correct parameters
        endpoint = "equipment/camera/warm"
        params = []
        
        if input.duration is not None:
            params.append(f"minutes={input.duration}")
        if input.cancel is not None:
            params.append(f"cancel={str(input.cancel).lower()}")
            
        if params:
            endpoint += "?" + "&".join(params)

        # Start warming
        result = await client._send_request("GET", endpoint)
        
        if not result.get("Success", False):
            return create_error_response(
                "NINACameraError",
                result.get("Error", "Unknown error starting camera warming"),
                {"StatusCode": result.get("StatusCode", 500)}
            )
        
        return {
            "Success": True,
            "Message": "Camera warming started successfully",
            "Details": result,
            "Type": "NINA_API"
        }

    except Exception as e:
        return create_error_response("NINACameraError", str(e))

class FlatPanelConnectInput(BaseModel):
    """Input model for flat panel connection settings"""
    device_id: Optional[str] = None

class FlatPanelLightInput(BaseModel):
    """Input model for flat panel light settings"""
    power: bool  # True to enable, False to disable

class FlatPanelCoverInput(BaseModel):
    """Input model for flat panel cover settings"""
    closed: bool  # True to close, False to open

class FlatPanelBrightnessInput(BaseModel):
    """Input model for flat panel brightness settings"""
    brightness: int  # Brightness value (0-100)

@mcp.tool()
async def nina_connect_flatpanel(input: FlatPanelConnectInput) -> Dict[str, Any]:
    """Connect to a flat panel device in NINA astronomy software.
    
    Args:
        input: FlatPanelConnectInput containing:
            device_id: Optional device ID to connect to
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatPanelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        endpoint = "equipment/flat-panel/connect"
        if input.device_id:
            endpoint += f"?to={input.device_id}"

        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": "Flat panel connected successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatPanelError", str(e))

@mcp.tool()
async def nina_disconnect_flatpanel() -> Dict[str, Any]:
    """Disconnect the flat panel from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatPanelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/flat-panel/disconnect")
        return {
            "Success": True,
            "Message": "Flat panel disconnected successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatPanelError", str(e))

@mcp.tool()
async def nina_list_flatpanel_devices() -> Dict[str, Any]:
    """List available flat panel devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatPanelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/flat-panel/list-devices")
        return {
            "Success": True,
            "Message": "Flat panel devices listed successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatPanelError", str(e))

@mcp.tool()
async def nina_rescan_flatpanel_devices() -> Dict[str, Any]:
    """Rescan for flat panel devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatPanelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/flat-panel/rescan")
        return {
            "Success": True,
            "Message": "Flat panel devices rescanned successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatPanelError", str(e))

@mcp.tool()
async def nina_get_flatpanel_info() -> Dict[str, Any]:
    """Get information about the connected flat panel in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Device information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatPanelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/flat-panel/info")
        return {
            "Success": True,
            "Message": "Flat panel information retrieved successfully",
            "Info": result.get("Response", {}),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatPanelError", str(e))

@mcp.tool()
async def nina_set_flatpanel_light(input: FlatPanelLightInput) -> Dict[str, Any]:
    """Set the flat panel light state in NINA astronomy software.
    
    Args:
        input: FlatPanelLightInput containing:
            power: True to enable, False to disable
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatPanelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", f"equipment/flat-panel/set-light?power={str(input.power).lower()}")
        return {
            "Success": True,
            "Message": f"Flat panel light {'enabled' if input.power else 'disabled'} successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatPanelError", str(e))

@mcp.tool()
async def nina_set_flatpanel_cover(input: FlatPanelCoverInput) -> Dict[str, Any]:
    """Set the flat panel cover state in NINA astronomy software.
    
    Args:
        input: FlatPanelCoverInput containing:
            closed: True to close, False to open
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatPanelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", f"equipment/flat-panel/set-cover?closed={str(input.closed).lower()}")
        return {
            "Success": True,
            "Message": f"Flat panel cover {'closed' if input.closed else 'opened'} successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatPanelError", str(e))

@mcp.tool()
async def nina_set_flatpanel_brightness(input: FlatPanelBrightnessInput) -> Dict[str, Any]:
    """Set the flat panel brightness in NINA astronomy software.
    
    Args:
        input: FlatPanelBrightnessInput containing:
            brightness: Brightness value (0-100)
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFlatPanelError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", f"equipment/flat-panel/set-brightness?brightness={input.brightness}")
        return {
            "Success": True,
            "Message": f"Flat panel brightness set to {input.brightness} successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFlatPanelError", str(e))

class FocuserConnectInput(BaseModel):
    """Input model for focuser connection settings"""
    device_id: Optional[str] = None

class FocuserMoveInput(BaseModel):
    """Input model for focuser movement settings"""
    position: int  # Target position in steps
    relative: Optional[bool] = None  # Whether the position is relative to current position

class FocuserTemperatureInput(BaseModel):
    """Input model for focuser temperature compensation settings"""
    enabled: bool  # True to enable, False to disable
    temperature: Optional[float] = None  # Target temperature in Celsius

@mcp.tool()
async def nina_connect_focuser(input: FocuserConnectInput) -> Dict[str, Any]:
    """Connect to a focuser device in NINA astronomy software.
    
    Args:
        input: FocuserConnectInput containing:
            device_id: Optional device ID to connect to
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFocuserError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        endpoint = "equipment/focuser/connect"
        if input.device_id:
            endpoint += f"?to={input.device_id}"

        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": "Focuser connected successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFocuserError", str(e))

@mcp.tool()
async def nina_disconnect_focuser() -> Dict[str, Any]:
    """Disconnect the focuser from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFocuserError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/focuser/disconnect")
        return {
            "Success": True,
            "Message": "Focuser disconnected successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFocuserError", str(e))

@mcp.tool()
async def nina_list_focuser_devices() -> Dict[str, Any]:
    """List available focuser devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFocuserError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/focuser/list-devices")
        return {
            "Success": True,
            "Message": "Focuser devices listed successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFocuserError", str(e))

@mcp.tool()
async def nina_rescan_focuser_devices() -> Dict[str, Any]:
    """Rescan for focuser devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFocuserError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/focuser/rescan")
        return {
            "Success": True,
            "Message": "Focuser devices rescanned successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFocuserError", str(e))

@mcp.tool()
async def nina_get_focuser_info() -> Dict[str, Any]:
    """Get information about the connected focuser in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Device information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFocuserError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/focuser/info")
        return {
            "Success": True,
            "Message": "Focuser information retrieved successfully",
            "Info": result.get("Response", {}),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFocuserError", str(e))

@mcp.tool()
async def nina_move_focuser(input: FocuserMoveInput) -> Dict[str, Any]:
    """Move the focuser to a specific position in NINA astronomy software.
    
    Args:
        input: FocuserMoveInput containing:
            position: Target position in steps
            relative: Optional whether the position is relative to current position
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFocuserError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        endpoint = f"equipment/focuser/move?position={input.position}"
        if input.relative is not None:
            endpoint += f"&relative={str(input.relative).lower()}"

        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": f"Focuser moved to position {input.position} successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFocuserError", str(e))

@mcp.tool()
async def nina_set_focuser_temperature(input: FocuserTemperatureInput) -> Dict[str, Any]:
    """Set the focuser temperature compensation in NINA astronomy software.
    
    Args:
        input: FocuserTemperatureInput containing:
            enabled: True to enable, False to disable
            temperature: Optional target temperature in Celsius
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAFocuserError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        endpoint = f"equipment/focuser/set-temperature?enabled={str(input.enabled).lower()}"
        if input.temperature is not None:
            endpoint += f"&temperature={input.temperature}"

        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": f"Focuser temperature compensation {'enabled' if input.enabled else 'disabled'} successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAFocuserError", str(e))

class ImageParameterInput(BaseModel):
    """Input model for image parameter settings"""
    parameter: str  # The parameter to get or set
    value: Optional[Any] = None  # The value to set (if setting a parameter)

@mcp.tool()
async def nina_get_image_parameter(input: ImageParameterInput) -> Dict[str, Any]:
    """Get an image parameter from NINA astronomy software.
    
    Args:
        input: ImageParameterInput containing:
            parameter: The parameter to get
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Parameter value
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAImageError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", f"image/parameter?parameter={input.parameter}")
        return {
            "Success": True,
            "Message": f"Image parameter {input.parameter} retrieved successfully",
            "Value": result.get("Response", {}),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAImageError", str(e))

@mcp.tool()
async def nina_set_image_parameter(input: ImageParameterInput) -> Dict[str, Any]:
    """Set an image parameter in NINA astronomy software.
    
    Args:
        input: ImageParameterInput containing:
            parameter: The parameter to set
            value: The value to set
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAImageError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        endpoint = f"image/parameter?parameter={input.parameter}"
        if input.value is not None:
            endpoint += f"&value={input.value}"

        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": f"Image parameter {input.parameter} set successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAImageError", str(e))

@mcp.tool()
async def nina_get_image_parameters() -> Dict[str, Any]:
    """Get all image parameters from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - All image parameters
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAImageError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "image/parameters")
        return {
            "Success": True,
            "Message": "Image parameters retrieved successfully",
            "Parameters": result.get("Response", {}),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAImageError", str(e))

@mcp.tool()
async def nina_reset_image_parameters() -> Dict[str, Any]:
    """Reset all image parameters to their default values in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAImageError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "image/parameters/reset")
        return {
            "Success": True,
            "Message": "Image parameters reset successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINAImageError", str(e))

class ImageGetInput(BaseModel):
    """Input model for getting an image"""
    index: int  # Index of the image to get
    resize: Optional[bool] = None  # Whether to resize the image
    quality: Optional[int] = None  # Quality of the image (1-100, -1 for png)
    size: Optional[str] = None  # Size of the image (widthxheight)
    scale: Optional[float] = None  # Scale of the image
    factor: Optional[float] = None  # Stretch factor (0-1)
    blackClipping: Optional[float] = None  # Black clipping value
    unlinked: Optional[bool] = None  # Whether stretch should be unlinked
    stream: Optional[bool] = None  # Whether to stream the image
    debayer: Optional[bool] = None  # Whether to debayer the image
    bayerPattern: Optional[str] = None  # Bayer pattern for debayering
    autoPrepare: Optional[bool] = None  # Whether to use NINA's processing
    imageType: Optional[str] = None  # Filter by image type (LIGHT, FLAT, DARK, BIAS, SNAPSHOT)
    save: Optional[bool] = None  # Whether to save the image to DEFAULT_IMAGE_SAVE_DIR
    filename: Optional[str] = None  # Optional filename for saved image

@mcp.tool()
async def nina_get_image(input: ImageGetInput) -> Dict[str, Any]:
    """Get an image from NINA astronomy software.
    
    Args:
        input: ImageGetInput containing:
            index: Index of the image to get
            resize: Whether to resize the image
            quality: Quality of the image (1-100, -1 for png)
            size: Size of the image (widthxheight)
            scale: Scale of the image
            factor: Stretch factor (0-1)
            blackClipping: Black clipping value
            unlinked: Whether stretch should be unlinked
            stream: Whether to stream the image
            debayer: Whether to debayer the image
            bayerPattern: What bayer pattern to use for debayering
            autoPrepare: Whether to use NINA's processing
            imageType: Filter by image type
            save: Whether to save the image
            filename: Optional filename for saved image
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Image data
        - Type of response
        - Saved path (if save=True)
    """
    try:
        client = await get_client()
        if not client._connected:
            try:
                await client.connect()
            except Exception as e:
                return create_error_response(
                    "NINAImageError",
                    str(e),
                    {"StatusCode": 401}
                )

        # Build the endpoint URL with query parameters
        endpoint = f"image/{input.index}"
        params = []
        
        # Add all optional parameters if they are provided
        if input.resize is not None:
            params.append(f"resize={str(input.resize).lower()}")
        if input.quality is not None:
            params.append(f"quality={input.quality}")
        if input.size:
            params.append(f"size={input.size}")
        if input.scale is not None:
            params.append(f"scale={input.scale}")
        if input.factor is not None:
            params.append(f"factor={input.factor}")
        if input.blackClipping is not None:
            params.append(f"blackClipping={input.blackClipping}")
        if input.unlinked is not None:
            params.append(f"unlinked={str(input.unlinked).lower()}")
        if input.stream is not None:
            params.append(f"stream={str(input.stream).lower()}")
        if input.debayer is not None:
            params.append(f"debayer={str(input.debayer).lower()}")
        if input.bayerPattern:
            params.append(f"bayerPattern={input.bayerPattern}")
        if input.autoPrepare is not None:
            params.append(f"autoPrepare={str(input.autoPrepare).lower()}")
        if input.imageType:
            params.append(f"imageType={input.imageType}")
        
        if params:
            endpoint += "?" + "&".join(params)

        # Set appropriate headers based on stream mode
        headers = {
            "Content-Type": "application/json"
        }
        if input.stream:
            headers["Accept"] = "image/jpeg, image/png"
        else:
            headers["Accept"] = "application/json"

        # Send request
        try:
            result = await client.session.get(
                f"http://{client.host}:{client.port}/v2/api/{endpoint}",
                headers=headers
            )
        except Exception as e:
            # Handle network/connection errors
            logger.error(f"Request failed: {str(e)}")
            return create_error_response(
                "NINAImageError",
            f"Failed to connect to NINA server: {str(e)}",
            {"StatusCode": 503}
            )
    
        # Handle non-200 responses with actual error from API
        if result.status != 200:
            error_text = await result.text()
            try:
                error_json = json.loads(error_text)
                error_msg = error_json.get("Error", error_text)
            except json.JSONDecodeError:
                error_msg = error_text
                
            logger.error(f"NINA API error: {error_msg}")
            return create_error_response(
                "NINAImageError",
                error_msg,
                {"StatusCode": result.status}
            )
            
        # Handle response based on stream mode
        if input.stream:
            # For stream mode, get binary data directly
            try:
                image_data = await result.read()
                content_type = result.headers.get('Content-Type', '')
                
                # Convert binary data to base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                response_data = {
                    "data": image_base64,
                    "content_type": content_type
                }
            except Exception as e:
                logger.error(f"Failed to process streamed image: {str(e)}")
                return create_error_response(
                    "NINAImageError",
                    f"Failed to process streamed image: {str(e)}",
                    {"StatusCode": 500}
                )
        else:
            # For non-stream mode, parse JSON response
            try:
                response_data = await result.json()
            except json.JSONDecodeError as e:
                error_text = await result.text()
                logger.error(f"Failed to parse JSON response: {str(e)}. Response text: {error_text}")
                return create_error_response(
                    "NINAImageError",
                    f"Invalid response from NINA server: {str(e)}",
                    {"StatusCode": 500}
                )
            
            # Check for API-level errors
            if not response_data.get("Success", False):
                error_msg = response_data.get("Error")
                status_code = response_data.get("StatusCode", 500)
                logger.error(f"NINA API error: {error_msg}")
                return create_error_response(
                    "NINAImageError",
                    error_msg,
                {"StatusCode": status_code}
                )
            
            response_data = response_data.get("Response", {})
        
        # Handle saving the image if requested
        saved_path = None
        if input.save:
            try:
                # Ensure the save directory exists
                os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
                
                # Generate filename if not provided
                if not input.filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ext = ".jpg" if input.stream else ".fits"  # Use appropriate extension
                    input.filename = f"nina_image_{input.index}_{timestamp}{ext}"
                
                # Save the image
                file_path = os.path.join(IMAGE_SAVE_DIR, input.filename)
                
                if input.stream:
                    # For stream mode, save binary data directly
                    with open(file_path, 'wb') as f:
                        f.write(image_data)
                else:
                    # For non-stream mode, handle potential base64 data
                    if isinstance(response_data, str):
                        image_bytes = base64.b64decode(response_data)
                    else:
                        image_bytes = response_data
                with open(file_path, 'wb') as f:
                    f.write(image_bytes)
                        
                saved_path = file_path
                logger.info(f"Image saved to: {saved_path}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
                return create_error_response(
                    "NINAImageError",
                    f"Failed to save image: {str(e)}",
                    {"StatusCode": 500}
                )
        
        return {
            "Success": True,
            "Message": "Image retrieved successfully",
            "Image": response_data,
            "Type": "NINA_API_STREAM" if input.stream else "NINA_API",
            "SavedPath": saved_path
        }
            
    except Exception as e:
        logger.error(f"Unexpected error in nina_get_image: {str(e)}")
        return create_error_response(
            "NINAImageError",
            str(e),
            {"StatusCode": 500}
        )

class ImageHistoryInput(BaseModel):
    """Input model for getting image history"""
    limit: Optional[int] = None  # Optional limit to number of images to return
    offset: Optional[int] = None  # Optional offset for pagination

class ImageThumbnailInput(BaseModel):
    """Input model for getting image thumbnail"""
    index: int  # Index of the image to get
    width: Optional[int] = None  # Optional width of thumbnail
    height: Optional[int] = None  # Optional height of thumbnail

@mcp.tool()
async def nina_get_image_history(input: ImageHistoryInput) -> Dict[str, Any]:
    """Get image history from NINA astronomy software.
    
    Args:
        input: ImageHistoryInput containing:
            limit: Optional limit to number of images to return
            offset: Optional offset for pagination
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Image history
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAImageError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # First check if the server is responding properly
        health_check = await client.session.get(
            f"http://{client.host}:{client.port}/v2/api"
        )
        if health_check.status != 200:
            return create_error_response(
                "NINAImageError",
                f"Server returned status {health_check.status}. Please ensure NINA is running and the API is enabled.",
                {"StatusCode": health_check.status}
            )

        # Construct the endpoint with parameters
        endpoint = "image-history"  # Changed from image/history to image-history
        params = []
        if input.limit is not None:
            params.append(f"limit={input.limit}")
        if input.offset is not None:
            params.append(f"offset={input.offset}")
        
        if params:
            endpoint += "?" + "&".join(params)

        # Get the image history with proper headers
        result = await client.session.get(
            f"http://{client.host}:{client.port}/v2/api/{endpoint}",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
        
        if result.status != 200:
            error_text = await result.text()
            logger.error(f"Failed to get image history. Server returned status {result.status}. Error: {error_text}")
            return create_error_response(
                "NINAImageError",
                f"Failed to get image history. Server returned status {result.status}. Error: {error_text}",
                {"StatusCode": result.status}
            )
            
        # Try to parse the response as JSON
        try:
            response = await result.json()
        except Exception as e:
            error_text = await result.text()
            logger.error(f"Failed to parse JSON response: {str(e)}. Response text: {error_text}")
            return create_error_response(
                "NINAImageError",
                f"Failed to parse JSON response: {str(e)}",
                {"StatusCode": 500}
            )
            
        if not response.get("Success"):
            error_msg = response.get("Error", "Unknown error from NINA")
            logger.error(f"NINA API error: {error_msg}")
            return create_error_response(
                "NINAImageError",
                error_msg,
                {"StatusCode": response.get("StatusCode", 500)}
            )
            
        # Get the total count of images
        image_count = len(response.get("Response", []))
        
        # Return the metadata for all images
        return {
            "Success": True,
            "Message": "Image history retrieved successfully",
            "Details": {
                "total_images": image_count,
                "images": [
                    {
                        "exposure_time": img.get("ExposureTime"),
                        "image_type": img.get("ImageType"),
                        "filter": img.get("Filter"),
                        "temperature": img.get("Temperature"),
                        "camera_name": img.get("CameraName"),
                        "gain": img.get("Gain"),
                        "offset": img.get("Offset"),
                        "date": img.get("Date"),
                        "telescope_name": img.get("TelescopeName"),
                        "focal_length": img.get("FocalLength"),
                        "stdev": img.get("StDev"),
                        "mean": img.get("Mean"),
                        "median": img.get("Median"),
                        "stars": img.get("Stars"),
                        "hfr": img.get("HFR"),
                        "is_bayered": img.get("IsBayered")
                    }
                    for img in response.get("Response", [])
                ]
            },
            "Type": "NINA_API"
        }
    except Exception as e:
        logger.error(f"Unexpected error in nina_get_image_history: {str(e)}")
        return create_error_response(
            "NINAImageError",
            str(e),
            {"StatusCode": 500}
        )

@mcp.tool()
async def nina_get_image_thumbnail(input: ImageThumbnailInput) -> Dict[str, Any]:
    """Get an image thumbnail from NINA astronomy software and save it to disk.
    
    Args:
        input: ImageThumbnailInput containing:
            index: Index of the image to get
            width: Optional width of thumbnail
            height: Optional height of thumbnail
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - File path where the thumbnail was saved
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINAImageError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        # Build the endpoint URL
        endpoint = f"image/thumbnail/{input.index}"
        params = []
        
        # Add optional size parameters
        if input.width is not None:
            params.append(f"width={input.width}")
        if input.height is not None:
            params.append(f"height={input.height}")
            
        if params:
            endpoint += "?" + "&".join(params)

        # Send request
        result = await client.session.get(
            f"http://{client.host}:{client.port}/v2/api/{endpoint}",
            headers={
                "Accept": "image/jpeg, image/png",
                "Content-Type": "application/json"
            }
        )
        
        if result.status != 200:
            error_text = await result.text()
            logger.error(f"Failed to get thumbnail. Server returned status {result.status}. Error: {error_text}")
            return create_error_response(
                "NINAImageError",
                f"Failed to get thumbnail. Server returned status {result.status}. Error: {error_text}",
                {"StatusCode": result.status}
            )
            
        # Ensure save directory exists
        thumbnail_dir = os.path.join(IMAGE_SAVE_DIR, "thumbnails")
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nina_thumbnail_{input.index}_{timestamp}.jpg"
        file_path = os.path.join(thumbnail_dir, filename)
        
        # Save the thumbnail
        image_data = await result.read()
        with open(file_path, 'wb') as f:
            f.write(image_data)
            
        logger.info(f"Thumbnail saved to: {file_path}")
        
        return {
            "Success": True,
            "Message": "Thumbnail retrieved and saved successfully",
            "FilePath": file_path,
            "FileName": filename,
            "Type": "NINA_API"
        }
    except Exception as e:
        logger.error(f"Error in nina_get_image_thumbnail: {str(e)}")
        return create_error_response("NINAImageError", str(e))

class RotatorConnectInput(BaseModel):
    """Input model for connecting to a rotator device"""
    device_id: Optional[str] = None  # Optional device ID to connect to

class RotatorMoveInput(BaseModel):
    """Input model for moving the rotator"""
    position: float  # Target position in degrees
    relative: Optional[bool] = None  # Whether the position is relative to current position

class RotatorSyncInput(BaseModel):
    """Input model for syncing the rotator"""
    position: float  # Position to sync to in degrees

class RotatorReverseInput(BaseModel):
    """Input model for setting rotator reverse state"""
    enabled: bool  # True to enable reverse, False to disable

@mcp.tool()
async def nina_connect_rotator(input: RotatorConnectInput) -> Dict[str, Any]:
    """Connect to a rotator device in NINA astronomy software.
    
    Args:
        input: RotatorConnectInput containing:
            device_id: Optional device ID to connect to
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINARotatorError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        endpoint = "equipment/rotator/connect"
        if input.device_id:
            endpoint += f"?to={input.device_id}"

        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": "Rotator connected successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINARotatorError", str(e))

@mcp.tool()
async def nina_disconnect_rotator() -> Dict[str, Any]:
    """Disconnect the rotator from NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINARotatorError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/rotator/disconnect")
        return {
            "Success": True,
            "Message": "Rotator disconnected successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINARotatorError", str(e))

@mcp.tool()
async def nina_list_rotator_devices() -> Dict[str, Any]:
    """List available rotator devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINARotatorError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/rotator/list-devices")
        return {
            "Success": True,
            "Message": "Rotator devices listed successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINARotatorError", str(e))

@mcp.tool()
async def nina_rescan_rotator_devices() -> Dict[str, Any]:
    """Rescan for rotator devices in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - List of available devices
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINARotatorError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/rotator/rescan")
        return {
            "Success": True,
            "Message": "Rotator devices rescanned successfully",
            "Devices": result.get("Response", []),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINARotatorError", str(e))

@mcp.tool()
async def nina_get_rotator_info() -> Dict[str, Any]:
    """Get information about the connected rotator in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Device information
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINARotatorError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/rotator/info")
        return {
            "Success": True,
            "Message": "Rotator information retrieved successfully",
            "Info": result.get("Response", {}),
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINARotatorError", str(e))

@mcp.tool()
async def nina_move_rotator(input: RotatorMoveInput) -> Dict[str, Any]:
    """Move the rotator to a specific position in NINA astronomy software.
    
    Args:
        input: RotatorMoveInput containing:
            position: Target position in degrees
            relative: Optional whether the position is relative to current position
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINARotatorError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        endpoint = f"equipment/rotator/move?position={input.position}"
        if input.relative is not None:
            endpoint += f"&relative={str(input.relative).lower()}"

        result = await client._send_request("GET", endpoint)
        return {
            "Success": True,
            "Message": f"Rotator moved to position {input.position}° successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINARotatorError", str(e))

@mcp.tool()
async def nina_halt_rotator() -> Dict[str, Any]:
    """Halt the rotator's current movement in NINA astronomy software.
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINARotatorError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", "equipment/rotator/halt")
        return {
            "Success": True,
            "Message": "Rotator movement halted successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINARotatorError", str(e))

@mcp.tool()
async def nina_sync_rotator(input: RotatorSyncInput) -> Dict[str, Any]:
    """Sync the rotator to a specific position in NINA astronomy software.
    
    Args:
        input: RotatorSyncInput containing:
            position: Position to sync to in degrees
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINARotatorError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", f"equipment/rotator/sync?position={input.position}")
        return {
            "Success": True,
            "Message": f"Rotator synced to position {input.position}° successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINARotatorError", str(e))

@mcp.tool()
async def nina_set_rotator_reverse(input: RotatorReverseInput) -> Dict[str, Any]:
    """Set the rotator's reverse state in NINA astronomy software.
    
    Args:
        input: RotatorReverseInput containing:
            enabled: True to enable reverse, False to disable
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Type of response
    """
    try:
        client = await get_client()
        if not client._connected:
            return create_error_response(
                "NINARotatorError",
                "Not connected to NINA server. Please connect first using nina_connect.",
                {"StatusCode": 401}
            )

        result = await client._send_request("GET", f"equipment/rotator/reverse?enabled={str(input.enabled).lower()}")
        return {
            "Success": True,
            "Message": f"Rotator reverse {'enabled' if input.enabled else 'disabled'} successfully",
            "Details": result,
            "Type": "NINA_API"
        }
    except Exception as e:
        return create_error_response("NINARotatorError", str(e))

class HelpInput(BaseModel):
    """Input model for getting help about specific tools"""
    tool_name: Optional[str] = None  # Optional tool name to get specific help
    category: Optional[str] = None  # Optional category to filter help content
    search: Optional[str] = None  # Optional search keyword to filter content
    list_tools: Optional[bool] = None  # List all available tools
    list_categories: Optional[bool] = None  # List all available categories

@mcp.tool()
async def nina_help(input: HelpInput) -> Dict[str, Any]:
    """Get comprehensive help about NINA tools and categories.
    
    Args:
        input: HelpInput containing:
            tool_name: Optional specific tool to get help for
            category: Optional category to filter help content
            search: Optional keyword to search across all help content
            list_tools: Optional flag to list all available tools
            list_categories: Optional flag to list all available categories
    
    Returns:
        Dict containing:
        - Success status
        - Message about the operation
        - Help content based on input filters
        - Type of response
    """
    try:
        # Read the help content from JSON file
        help_file_path = os.path.join(os.path.dirname(__file__), 'nina_help.json')
        try:
            with open(help_file_path, 'r') as f:
                help_content = json.load(f)
        except FileNotFoundError:
            return create_error_response(
                "NINAHelpError",
                f"Help file not found at {help_file_path}",
                {"StatusCode": 404}
            )
        except json.JSONDecodeError as e:
            return create_error_response(
                "NINAHelpError",
                f"Invalid JSON in help file: {str(e)}",
                {"StatusCode": 500}
            )

        # Extract categories and tool help from loaded content
        categories = help_content.get('help_categories', {})
        tool_help = help_content.get('tool_help', {})

        # List all tools if requested
        if input.list_tools:
            tools_list = []
            for tool_name, tool_info in tool_help.items():
                tools_list.append({
                    "name": tool_name,
                    "title": tool_info['title'],
                    "description": tool_info['description'],
                    "category": next((cat for cat, cat_info in categories.items() 
                                   if any(tool_name in tool for tool in cat_info.get('tools_overview', []))), None)
                })
            return {
                "Success": True,
                "Message": "List of all available tools",
                "Tools": tools_list,
                "Type": "NINA_API"
            }

        # List all categories if requested
        if input.list_categories:
            categories_list = []
            for cat_id, cat_info in categories.items():
                category_tools = []
                if cat_id == 'start_here':
                    # For start_here, get tools from tools_overview
                    for tool_overview in cat_info.get('tools_overview', []):
                        # Extract tool names from the overview text
                        tool_names = [name.strip() for name in tool_overview.split(':')[1].split(',')]
                        category_tools.extend(tool_names)
                else:
                    # For other categories, find tools that belong to this category
                    category_tools = [tool_name for tool_name, tool_info in tool_help.items()
                                    if any(cat_id in str(value).lower() for value in tool_info.values())]
                
                categories_list.append({
                    "id": cat_id,
                    "title": cat_info['title'],
                    "description": cat_info['description'],
                    "tools_count": len(category_tools),
                    "tools": category_tools
                })
            return {
                "Success": True,
                "Message": "List of all available categories",
                "Categories": categories_list,
                "Type": "NINA_API"
            }

        # If search keyword is provided, filter both categories and tools
        if input.search:
            search_term = input.search.lower()
            
            # Filter categories
            filtered_categories = {}
            for cat_id, cat_info in categories.items():
                # Search in title, description, and other fields
                if (search_term in cat_id.lower() or
                    search_term in cat_info['title'].lower() or
                    search_term in cat_info['description'].lower() or
                    (cat_id == 'start_here' and any(search_term in tool.lower() for tool in cat_info.get('tools_overview', [])))):
                    filtered_categories[cat_id] = cat_info

            # Filter tools
            filtered_tools = {}
            for tool_name, tool_info in tool_help.items():
                # Search in tool name, title, description, parameters, and examples
                if (search_term in tool_name.lower() or
                    search_term in tool_info['title'].lower() or
                    search_term in tool_info['description'].lower() or
                    any(search_term in str(param).lower() for param in tool_info.get('parameters', {}).values()) or
                    any(search_term in str(example).lower() for example in tool_info.get('examples', []))):
                    filtered_tools[tool_name] = tool_info

            return {
                "Success": True,
                "Message": f"Search results for: {input.search}",
                "Categories": filtered_categories,
                "Tools": filtered_tools,
                "Type": "NINA_API"
            }
        
        # If specific tool is requested
        if input.tool_name:
            if input.tool_name in tool_help:
                return {
                    "Success": True,
                    "Message": f"Help for tool: {input.tool_name}",
                    "Help": tool_help[input.tool_name],
                    "Type": "NINA_API"
                }
            return create_error_response(
                "NINAHelpError",
                f"Tool '{input.tool_name}' not found in help content",
                {"StatusCode": 404}
            )
        
        # If specific category is requested
        if input.category:
            if input.category in categories:
                return {
                    "Success": True,
                    "Message": f"Help for category: {input.category}",
                    "Category": categories[input.category],
                    "Type": "NINA_API"
                }
            return create_error_response(
                "NINAHelpError",
                f"Category '{input.category}' not found in help content",
                {"StatusCode": 404}
            )
        
        # Return all help content if no filters provided
        return {
            "Success": True,
            "Message": "Complete NINA MCP help content",
            "Categories": categories,
            "Tools": tool_help,
            "Type": "NINA_API"
        }

    except Exception as e:
        logger.error(f"Error in nina_help: {str(e)}")
        return create_error_response("NINAHelpError", str(e))

if __name__ == "__main__":
    logger.info("Starting NINA Advanced API Server...")
    logger.info(f"Connecting to NINA at {DEFAULT_NINA_HOST}:{DEFAULT_NINA_PORT}")
    
    # Run the FastMCP server
    mcp.run() 