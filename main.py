# main.py
#   main entry point
# by: Noah Syrkis

# Imports
import labs
import mlxp
from functools import reduce


@mlxp.launch(config_path="./conf")
def main(ctx: mlxp.Context) -> None:
    cfg = getattr(ctx.config, ctx.config.task)
    fn = reduce(getattr, (ctx.config.task, "main"), labs)
    fn(cfg) if callable(fn) else print("See that fn is in __init__")


if __name__ == "__main__":
    main()
