use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use ai2npu::cli::{Cli, Command};
use ai2npu::config::AppConfig;
use ai2npu::http::serve;
use ai2npu::openvino_backend::OpenVinoStatus;
use ai2npu::windows_service;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run { config } => {
            let cfg = AppConfig::load(config)?;
            configure_runtime_environment(&cfg)?;
            ai2npu::logs::init_file_logging(&cfg.logging)?;
            tracing::info!("starting ai2npu console host");
            let openvino = OpenVinoStatus::detect();
            tokio::runtime::Runtime::new()?.block_on(serve(cfg, openvino))?;
        }
        Command::RunService { config } => {
            windows_service::run_service(config)?;
        }
        Command::InstallService { config, exe } => {
            windows_service::install_service(exe, config)?;
            println!("service installed: {}", windows_service::SERVICE_NAME);
        }
        Command::InitConfig { path, data_dir } => {
            windows_service::init_config(path, data_dir)?;
        }
        Command::InstallModel { model, model_dir } => {
            ai2npu::model_installer::install_model(&model, &model_dir)?;
            println!("model installed: {model} -> {}", model_dir.display());
        }
        Command::UninstallService => {
            windows_service::uninstall_service()?;
            println!("service uninstalled: {}", windows_service::SERVICE_NAME);
        }
        Command::StartService => {
            windows_service::start_service()?;
            println!("service started: {}", windows_service::SERVICE_NAME);
        }
        Command::StopService => {
            windows_service::stop_service()?;
            println!("service stopped: {}", windows_service::SERVICE_NAME);
        }
        Command::RestartService => {
            windows_service::restart_service()?;
            println!("service restarted: {}", windows_service::SERVICE_NAME);
        }
        Command::ValidateConfig { config } => {
            let path = config.unwrap_or_else(default_config_path);
            AppConfig::load(&path)?;
            println!("config valid: {}", path.display());
        }
        Command::CheckNpu => {
            let status = OpenVinoStatus::detect();
            println!("runtime_available: {}", status.runtime_available);
            println!("devices: {}", status.devices.join(","));
            println!("npu_available: {}", status.npu_available);
            if let Some(error) = status.error {
                println!("error: {error}");
            }
        }
        Command::ListModels { config } => {
            let path = config.unwrap_or_else(default_config_path);
            let cfg = AppConfig::load(path)?;
            for model in cfg.models {
                println!(
                    "{}\t{}\t{}",
                    model.id,
                    model.model_type,
                    if model.enabled { "enabled" } else { "disabled" }
                );
            }
        }
        Command::Version => {
            println!("ai2npu {}", env!("CARGO_PKG_VERSION"));
        }
    }

    Ok(())
}

fn default_config_path() -> PathBuf {
    PathBuf::from(r"C:\ProgramData\ai2npu\config.toml")
}

fn configure_runtime_environment(cfg: &AppConfig) -> Result<()> {
    if std::env::var_os("AI2NPU_WHISPER_CACHE_DIR").is_none() {
        let cache_dir = cfg
            .logging
            .directory
            .parent()
            .map(|parent| parent.join("cache").join("whisper"))
            .unwrap_or_else(|| PathBuf::from("cache").join("whisper"));
        std::fs::create_dir_all(&cache_dir)?;
        std::env::set_var("AI2NPU_WHISPER_CACHE_DIR", cache_dir);
    }
    Ok(())
}
