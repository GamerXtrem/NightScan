"""Authentication module for NightScan"""

from .jwt_manager import (
    JWTManager,
    get_jwt_manager,
    jwt_required,
    jwt_optional,
    require_roles,
    TokenPayload
)

__all__ = [
    'JWTManager',
    'get_jwt_manager',
    'jwt_required',
    'jwt_optional',
    'require_roles',
    'TokenPayload'
]