"""训练计时上限的纯函数判定。

规则：`max_seconds = 0` 表示不设时间上限，任何耗时都不触发退出；
负值在命令行参数校验阶段即被拒绝。计时仍写入日志，只作记录。
"""


def time_limit_reached(max_seconds: int, elapsed_seconds: float) -> bool:
    """epoch 间与 epoch 内两处退出判定共用本函数，保证语义一致。"""
    return max_seconds > 0 and elapsed_seconds >= max_seconds
