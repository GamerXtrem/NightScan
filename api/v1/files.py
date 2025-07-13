"""API v1 - Files management endpoints with RESTful design."""

import logging
from flask import Blueprint, request, jsonify, send_file
from api_versioning.decorators import api_version, version_required
from api_versioning.registry import register_endpoint

logger = logging.getLogger(__name__)


def create_files_blueprint() -> Blueprint:
    """Create files blueprint for API v1."""
    files_bp = Blueprint("files_v1", __name__)

    # Import file handling functions if available
    try:
        from web.app import validate_wav_signature, sanitize_filename, MAX_FILE_SIZE
        files_available = True
    except ImportError:
        logger.warning("File handling functions not available, using placeholder endpoints")
        files_available = False

    # Register RESTful endpoints in the registry
    endpoints = [
        # RESTful endpoints
        {
            "path": "/api/v1/files",
            "methods": ["POST"],
            "description": "Upload file for processing",
            "tags": ["files", "upload"],
        },
        {
            "path": "/api/v1/files/<file_id>",
            "methods": ["GET"],
            "description": "Download or get file information",
            "tags": ["files", "download"],
        },
        {
            "path": "/api/v1/files/<file_id>",
            "methods": ["DELETE"],
            "description": "Delete uploaded file",
            "tags": ["files", "delete"],
        },
        {
            "path": "/api/v1/files",
            "methods": ["GET"],
            "description": "List uploaded files",
            "tags": ["files", "list"],
        },
        {
            "path": "/api/v1/files/statistics",
            "methods": ["GET"],
            "description": "Get file storage statistics",
            "tags": ["files", "statistics"],
        },
        # Legacy endpoints for backward compatibility
        {
            "path": "/api/v1/files/upload",
            "methods": ["POST"],
            "description": "[DEPRECATED] Use POST /api/v1/files instead",
            "tags": ["files", "deprecated"],
        },
        {
            "path": "/api/v1/files/download/<file_id>",
            "methods": ["GET"],
            "description": "[DEPRECATED] Use GET /api/v1/files/<file_id> instead",
            "tags": ["files", "deprecated"],
        },
    ]

    for endpoint in endpoints:
        register_endpoint(
            path=endpoint["path"],
            methods=endpoint["methods"],
            version="v1",
            description=endpoint["description"],
            tags=endpoint["tags"],
            requires_auth=True,
        )

    # RESTful routes with proper HTTP methods

    @files_bp.route("", methods=["POST"])
    @api_version("v1", description="Upload file", tags=["files"])
    @version_required(min_version="v1")
    def create_file():
        """Upload a file (RESTful)."""
        if not request.files:
            return jsonify({
                "error": "No file provided",
                "message": "Please provide a file to upload"
            }), 400

        file = request.files.get("file")
        if not file:
            return jsonify({
                "error": "No file provided",
                "message": "File field is required"
            }), 400

        # Optional metadata
        description = request.form.get("description", "")
        category = request.form.get("category", "general")

        if files_available:
            try:
                # Validate file
                filename = sanitize_filename(file.filename)
                
                # Check file size
                if len(file.read()) > MAX_FILE_SIZE:
                    return jsonify({
                        "error": "File too large",
                        "message": f"File size must be less than {MAX_FILE_SIZE} bytes"
                    }), 413
                
                file.seek(0)  # Reset file pointer
                
                # For WAV files, validate signature
                if filename.lower().endswith('.wav'):
                    if not validate_wav_signature(file):
                        return jsonify({
                            "error": "Invalid WAV file",
                            "message": "File signature validation failed"
                        }), 400
                    file.seek(0)

                # Save file (placeholder logic)
                file_id = "file_123456"
                file_path = f"/uploads/{filename}"
                
                return jsonify({
                    "file_id": file_id,
                    "filename": filename,
                    "size": len(file.read()),
                    "category": category,
                    "description": description,
                    "status": "uploaded",
                    "upload_url": f"/api/v1/files/{file_id}",
                    "api_version": "v1"
                }), 201

            except Exception as e:
                logger.error(f"File upload error: {e}")
                return jsonify({
                    "error": "Upload failed",
                    "message": str(e)
                }), 500
        else:
            # Return mock response
            return jsonify({
                "file_id": "mock_file_123",
                "filename": file.filename,
                "size": 1024,
                "category": category,
                "description": description,
                "status": "uploaded",
                "upload_url": "/api/v1/files/mock_file_123",
                "api_version": "v1"
            }), 201

    @files_bp.route("", methods=["GET"])
    @api_version("v1", description="List files", tags=["files"])
    @version_required(min_version="v1")
    def list_files():
        """List uploaded files (RESTful)."""
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        category = request.args.get("category")
        
        # Mock file list
        files = [
            {
                "file_id": "file_001",
                "filename": "recording1.wav",
                "size": 2048,
                "category": "audio",
                "uploaded_at": "2024-01-15T10:30:00Z",
                "status": "processed"
            },
            {
                "file_id": "file_002", 
                "filename": "image1.jpg",
                "size": 4096,
                "category": "image",
                "uploaded_at": "2024-01-15T09:15:00Z",
                "status": "processing"
            }
        ]
        
        # Filter by category if provided
        if category:
            files = [f for f in files if f["category"] == category]
        
        return jsonify({
            "files": files,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": len(files),
                "has_next": False,
                "has_prev": False
            },
            "api_version": "v1"
        }), 200

    @files_bp.route("/<file_id>", methods=["GET"])
    @api_version("v1", description="Get file", tags=["files"])
    @version_required(min_version="v1")
    def get_file(file_id: str):
        """Get file information or download file (RESTful)."""
        action = request.args.get("action", "info")
        
        if action == "download":
            # Return file download
            try:
                # In real implementation, get file from storage
                return jsonify({
                    "error": "Not implemented",
                    "message": "File download functionality is being implemented"
                }), 501
            except Exception as e:
                logger.error(f"File download error: {e}")
                return jsonify({
                    "error": "Download failed",
                    "message": str(e)
                }), 500
        else:
            # Return file information
            return jsonify({
                "file_id": file_id,
                "filename": "example.wav",
                "size": 2048,
                "category": "audio",
                "description": "Audio recording",
                "uploaded_at": "2024-01-15T10:30:00Z",
                "status": "processed",
                "download_url": f"/api/v1/files/{file_id}?action=download",
                "api_version": "v1"
            }), 200

    @files_bp.route("/<file_id>", methods=["DELETE"])
    @api_version("v1", description="Delete file", tags=["files"])
    @version_required(min_version="v1")
    def delete_file(file_id: str):
        """Delete uploaded file (RESTful)."""
        try:
            # In real implementation, delete from storage
            return jsonify({
                "file_id": file_id,
                "status": "deleted",
                "message": "File successfully deleted",
                "api_version": "v1"
            }), 200
        except Exception as e:
            logger.error(f"File deletion error: {e}")
            return jsonify({
                "error": "Deletion failed",
                "message": str(e)
            }), 500

    @files_bp.route("/statistics", methods=["GET"])
    @api_version("v1", description="File statistics", tags=["files"])
    @version_required(min_version="v1")
    def get_file_statistics():
        """Get file storage statistics."""
        return jsonify({
            "total_files": 156,
            "total_size_bytes": 1048576000,  # 1GB
            "storage_used_gb": 1.0,
            "storage_limit_gb": 10.0,
            "files_by_category": {
                "audio": 89,
                "image": 45,
                "video": 22
            },
            "files_by_status": {
                "uploaded": 12,
                "processing": 8,
                "processed": 134,
                "error": 2
            },
            "api_version": "v1"
        }), 200

    # Legacy endpoints (deprecated)

    @files_bp.route("/upload", methods=["POST"])
    @api_version("v1", description="[DEPRECATED] Upload file", tags=["files", "deprecated"])
    @version_required(min_version="v1")
    def upload_file_legacy():
        """Upload file (deprecated)."""
        response = create_file()
        if hasattr(response, 'headers'):
            response.headers['X-API-Deprecation-Warning'] = 'This endpoint is deprecated. Use POST /api/v1/files instead.'
            response.headers['X-API-Deprecated-Endpoint'] = '/api/v1/files/upload'
            response.headers['X-API-Replacement-Endpoint'] = '/api/v1/files'
        return response

    @files_bp.route("/download/<file_id>", methods=["GET"])
    @api_version("v1", description="[DEPRECATED] Download file", tags=["files", "deprecated"])
    @version_required(min_version="v1")
    def download_file_legacy(file_id: str):
        """Download file (deprecated)."""
        response = get_file(file_id)
        if hasattr(response, 'headers'):
            response.headers['X-API-Deprecation-Warning'] = 'This endpoint is deprecated. Use GET /api/v1/files/{file_id}?action=download instead.'
            response.headers['X-API-Deprecated-Endpoint'] = f'/api/v1/files/download/{file_id}'
            response.headers['X-API-Replacement-Endpoint'] = f'/api/v1/files/{file_id}?action=download'
        return response

    return files_bp