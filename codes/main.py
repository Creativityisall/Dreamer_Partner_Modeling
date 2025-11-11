import elements  # 导入自定义的工具模块，可能包含文件路径、日志打印、计时器等功能
from trainers.dreamer_trainer import DreamerTrainer  # 导入 Dreamer 算法的训练器
from trainers.on_policy_trainer import OnPolicyTrainer  # 导入 On-Policy 算法的训练器
from utils.tools import set_seed, get_task_name  # 导入工具函数，用于设置随机种子和获取任务名称

import ruamel.yaml as yaml  # 导入 YAML 解析库
import sys
from typing import List


def main(argv: List[str]):
    """
    主函数：解析配置，设置环境和训练器，并启动训练。

    Args:
        argv (List[str]): 命令行参数列表（通常是 sys.argv[1:]）。
    """
    # 打印一个自定义的 ASCII 艺术标题（可能是项目或框架名称）
    elements.print(r"---   ____  __  __    ___        ____  __  ---")
    elements.print(r"---  |  _ \|  \/  |  / \ \      / /  \/  | ---")
    elements.print(r"---  | | | | |\/| | / _ \ \ /\ / /| |\/| | ---")
    elements.print(r"---  | |_| | |  | |/ ___ \ V  V / | |  | | ---")
    elements.print(r"---  |____/|_|  |_/_/   \_\_/\_/  |_|  |_| ---")

    # 1. 命令行配置解析 (Flags)

    # 首先，解析命令行中关于环境和训练器的基础参数，并获取剩余参数
    # 默认值：env="smac", trainer="imagination" (可能用于加载默认的配置文件)
    config, remaining = elements.Flags(env="smac", trainer="dreamer").parse_known(argv)

    # 2. 加载 Trainer 特定配置

    # 构造并读取训练器对应的 YAML 配置文件
    trainer_config_path = elements.Path(__file__).parent / 'configs' / 'trainer_configs' / f'{config.trainer}.yaml'
    trainer_config = yaml.YAML(typ='safe').load(trainer_config_path.read())
    # 将训练器配置更新到主配置中
    config = config.update(trainer_config)

    # 3. 加载 Environment 特定配置

    # 构造并读取环境对应的 YAML 配置文件
    env_config_path = elements.Path(__file__).parent / 'configs' / 'env_configs' / f'{config.env}.yaml'
    env_config = yaml.YAML(typ='safe').load(env_config_path.read())
    # 将环境配置更新到主配置中
    config = config.update(env_config)

    # 4. 再次解析命令行参数

    # 使用完整的配置结构，再次解析剩余的命令行参数，允许用户覆盖 YAML 中的任何设置
    config = elements.Flags(config).parse(remaining)

    # 5. 设置日志目录并保存配置

    # 生成任务名称（基于配置中的关键参数）
    task_name = get_task_name(config)
    # 定义完整的日志目录路径：logs/env/task_name/config_name/timestamp/
    logdir = elements.Path(__file__).parent / 'logs' / config.env / task_name / config.name / elements.timestamp()
    # 将日志目录路径保存回配置对象
    config = config.update(logdir=logdir)
    # 创建日志目录
    logdir.mkdir()
    # 将最终的配置保存到日志目录中
    config.save(logdir / 'config.yaml')

    # 6. 设置随机种子
    set_seed(config)

    # 7. 设置计时器
    # 根据配置启用或禁用全局计时器
    elements.timer.global_timer.enabled = config.logging.timer

    # 8. 启动训练

    elements.print(f"Starting training for trainer: {config.trainer}")

    # 根据配置中的 trainer 类型实例化对应的训练器
    if config.trainer == "dreamer":
        trainer = DreamerTrainer(config)
    elif config.trainer == "on_policy":
        trainer = OnPolicyTrainer(config)
    else:
        # 如果 trainer 类型不支持，则抛出错误
        raise ValueError(f"Trainer {config.trainer} not found")

    # 启动训练循环
    trainer.train()


if __name__ == "__main__":
    # 当脚本直接运行时，从系统参数中获取除脚本名外的所有参数
    main(sys.argv[1:])