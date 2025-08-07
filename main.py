import labs
import mlxp
from functools import reduce
from typing import


@mlxp.launch(config_path="./conf")
def main(ctx: mlxp.Context) -> None:
    fn = reduce(getattr, ctx.config.task.split("."), labs)
    fn(ctx.config) if callable(fn) else print("See that fn is in __init__")


if __name__ == "__main__":
    main()
