import secrets

# Generates a secure 64-character hexadecimal string suitable for JWT HS256
jwt_secret = secrets.token_hex(32)
print(f"JWT_SECRET_KEY={jwt_secret}")