import os
from flask import Flask

def create_app(test_config=None):
    """
    Application factory for the ORBIT platform.
    Initializes configuration, static/template routes, and registers blueprints.
    """
    app = Flask(__name__, instance_relative_config=True)

    # Base configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "orbit-cyber-telemetry-key-2026"),
        UPLOAD_FOLDER=os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "uploads"),
        MAX_CONTENT_LENGTH=32 * 1024 * 1024,  # 32MB total payload limit
        ALLOWED_EXTENSIONS={"png", "jpg", "jpeg", "tif", "tiff"},
        MIN_DIMENSION=256,
        MAX_DIMENSION=4096,
    )

    if test_config:
        app.config.update(test_config)

    # Ensure upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Register routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app

# Expose app at module level for gunicorn "app:app" default start command
app = create_app()
