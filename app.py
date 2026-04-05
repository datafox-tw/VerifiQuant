import os
# old, not available now
from verifiquant_v1.app import create_app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.environ.get("PORT", 6222)), debug=True)
