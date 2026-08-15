import logging
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from rich import print
from rich.syntax import Syntax
from rich.tree import Tree

log = logging.getLogger(__name__)


def extras(config: DictConfig) -> None:
    

    
    OmegaConf.set_struct(config, False)

    
    if config.get("disable_warnings"):
        log.info(f"Disabling python warnings! <config.disable_warnings=True>")
        warnings.filterwarnings("ignore")

    
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    
    if config.trainer.get("accelerator") in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info("Forcing ddp friendly configuration! <config.trainer.accelerator=ddp>")
        
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    
    OmegaConf.set_struct(config, True)


def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    

    style = "dim"
    tree = Tree(f":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    print(tree)


def empty(*args, **kwargs):
    pass


def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    

    hparams = {}

    
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "optimizer" in config:
        hparams["optimizer"] = config["optimizer"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    
    
    
    
    
    
    
    
    

    
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    
    trainer.logger.log_hyperparams(hparams)

    
    
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    

    
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb.finish()
