import labs
import mlxp
from functools import reduce


@mlxp.launch(config_path="./conf")
def main(ctx: mlxp.Context) -> None:
    cfg, _ = ctx.config, ctx.logger
    fn = reduce(getattr, cfg.task.split("."), labs)
    assert callable(fn), f"Expected {cfg.task} to be callable"
    fn(cfg)


if __name__ == "__main__":
    main()
