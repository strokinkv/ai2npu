use std::path::PathBuf;

use ai2npu::cli::{Cli, Command};
use ai2npu::windows_service::ServiceInstallPlan;
use clap::Parser;

#[test]
fn parses_windows_service_commands() {
    assert!(matches!(
        Cli::parse_from([
            "ai2npu",
            "install-service",
            "--config",
            r"C:\ProgramData\ai2npu\config.toml"
        ])
        .command,
        Command::InstallService { .. }
    ));
    assert!(matches!(
        Cli::parse_from(["ai2npu", "uninstall-service"]).command,
        Command::UninstallService
    ));
    assert!(matches!(
        Cli::parse_from(["ai2npu", "start-service"]).command,
        Command::StartService
    ));
    assert!(matches!(
        Cli::parse_from(["ai2npu", "stop-service"]).command,
        Command::StopService
    ));
    assert!(matches!(
        Cli::parse_from(["ai2npu", "restart-service"]).command,
        Command::RestartService
    ));
}

#[test]
fn install_service_accepts_config_and_exe_path() {
    let cli = Cli::parse_from([
        "ai2npu",
        "install-service",
        "--config",
        r"C:\ProgramData\ai2npu\config.toml",
        "--exe",
        r"C:\Program Files\ai2npu\ai2npu.exe",
    ]);

    let Command::InstallService { config, exe } = cli.command else {
        panic!("expected install-service command");
    };
    assert_eq!(config, PathBuf::from(r"C:\ProgramData\ai2npu\config.toml"));
    assert_eq!(
        exe,
        Some(PathBuf::from(r"C:\Program Files\ai2npu\ai2npu.exe"))
    );
}

#[test]
fn parses_hidden_run_service_command() {
    let cli = Cli::parse_from([
        "ai2npu",
        "run-service",
        "--config",
        r"C:\ProgramData\ai2npu\config.toml",
    ]);

    let Command::RunService { config } = cli.command else {
        panic!("expected run-service command");
    };
    assert_eq!(config, PathBuf::from(r"C:\ProgramData\ai2npu\config.toml"));
}

#[test]
fn service_install_plan_uses_run_service_entrypoint() {
    let plan = ServiceInstallPlan::new(
        PathBuf::from(r"C:\Program Files\ai2npu\ai2npu.exe"),
        PathBuf::from(r"C:\ProgramData\ai2npu\config.toml"),
    );

    assert_eq!(
        plan.launch_arguments,
        vec![
            "run-service",
            "--config",
            r"C:\ProgramData\ai2npu\config.toml"
        ]
    );
}
