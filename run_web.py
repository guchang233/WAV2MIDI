from audiomidi_app.cloud_server import main
import sys

if __name__ == "__main__":
    if "--frontend" not in sys.argv:
        sys.argv.append("--frontend")
    raise SystemExit(main())
