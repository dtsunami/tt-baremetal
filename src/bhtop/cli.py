"""bhtop entry point: terminal-graphics floorplan by default, Rich heatmap with --simple."""
import sys


def main():
    argv = sys.argv[1:]
    if "--simple" in argv:
        sys.argv = [sys.argv[0]] + [a for a in argv if a != "--simple"]
        from .app import main as simple_main
        return simple_main()
    from .textual_app import main as gfx_main
    return gfx_main()


if __name__ == "__main__":
    main()
