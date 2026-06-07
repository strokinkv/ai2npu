use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "ai2npu")]
#[command(about = "Local OpenAI-compatible OpenVINO/NPU service")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Run {
        #[arg(long)]
        config: PathBuf,
    },
    #[command(hide = true)]
    RunService {
        #[arg(long)]
        config: PathBuf,
    },
    InstallService {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        exe: Option<PathBuf>,
    },
    #[command(hide = true)]
    InitConfig {
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        data_dir: PathBuf,
    },
    InstallModel {
        #[arg(long)]
        model: String,
        #[arg(long)]
        model_dir: PathBuf,
    },
    UninstallService,
    StartService,
    StopService,
    RestartService,
    ValidateConfig {
        #[arg(long)]
        config: Option<PathBuf>,
    },
    CheckNpu,
    ListModels {
        #[arg(long)]
        config: Option<PathBuf>,
    },
    Version,
}
