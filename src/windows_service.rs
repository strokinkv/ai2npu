use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, Result};

pub const SERVICE_NAME: &str = "ai2npuService";
pub const SERVICE_DISPLAY_NAME: &str = "ai2npu Service";
pub const SERVICE_DESCRIPTION: &str = "Local OpenAI-compatible OpenVINO/NPU service";
static SERVICE_CONFIG: OnceLock<PathBuf> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceInstallPlan {
    pub service_name: &'static str,
    pub display_name: &'static str,
    pub executable: PathBuf,
    pub config: PathBuf,
    pub launch_arguments: Vec<String>,
}

impl ServiceInstallPlan {
    pub fn new(executable: PathBuf, config: PathBuf) -> Self {
        Self {
            service_name: SERVICE_NAME,
            display_name: SERVICE_DISPLAY_NAME,
            executable,
            config: config.clone(),
            launch_arguments: vec![
                "run-service".to_string(),
                "--config".to_string(),
                config.display().to_string(),
            ],
        }
    }
}

pub fn install_service(exe: Option<PathBuf>, config: PathBuf) -> Result<()> {
    let executable = exe.unwrap_or(std::env::current_exe()?);
    install_service_impl(ServiceInstallPlan::new(executable, config))
}

pub fn init_config(path: PathBuf, data_dir: PathBuf) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    let model_root = data_dir.join("models");
    let mut text = String::from(
        "[server]\r\n\
host = \"127.0.0.1\"\r\n\
port = 9555\r\n\
request_body_limit_mb = 100\r\n\
thread_count = 16\r\n\r\n\
[queue]\r\n\
max_pending_requests = 10\r\n\
default_timeout_sec = 600\r\n\r\n\
[logging]\r\n\
level = \"info\"\r\n",
    );
    text.push_str(&format!("directory = '{}\\logs'\r\n", data_dir.display()));
    text.push_str("max_file_size_mb = 10\r\nmax_files = 10\r\n");

    if model_root.join("strokinkv\\bge-m3-int8-ov").is_dir() {
        text.push_str(
            "\r\n[[models]]\r\n\
id = \"BAAI/bge-m3\"\r\n\
type = \"embedding\"\r\n",
        );
        text.push_str(&format!(
            "path = '{}\\strokinkv\\bge-m3-int8-ov'\r\n",
            model_root.display()
        ));
        text.push_str(
            "enabled = true\r\n\
preload = false\r\n\
queue_timeout_sec = 600\r\n\
normalize = true\r\n",
        );
    }

    if model_root
        .join("OpenVINO\\whisper-large-v3-turbo-int8-ov")
        .is_dir()
    {
        text.push_str(
            "\r\n[[models]]\r\n\
id = \"openai/whisper-large-v3-turbo\"\r\n\
type = \"whisper\"\r\n",
        );
        text.push_str(&format!(
            "path = '{}\\OpenVINO\\whisper-large-v3-turbo-int8-ov'\r\n",
            model_root.display()
        ));
        text.push_str(
            "enabled = true\r\n\
preload = false\r\n\
queue_timeout_sec = 600\r\n\
max_audio_duration_sec = 1800\r\n",
        );
    }

    std::fs::write(&path, text)
        .with_context(|| format!("failed to write config {}", path.display()))?;
    Ok(())
}

pub fn uninstall_service() -> Result<()> {
    uninstall_service_impl()
}

pub fn start_service() -> Result<()> {
    start_service_impl()
}

pub fn stop_service() -> Result<()> {
    stop_service_impl()
}

pub fn restart_service() -> Result<()> {
    stop_service()?;
    start_service()
}

pub fn run_service(config: PathBuf) -> Result<()> {
    run_service_impl(config)
}

#[cfg(windows)]
fn install_service_impl(plan: ServiceInstallPlan) -> Result<()> {
    use std::ffi::OsString;
    use windows_service::service::{
        ServiceAccess, ServiceErrorControl, ServiceInfo, ServiceStartType, ServiceType,
    };
    use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

    let manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CREATE_SERVICE | ServiceManagerAccess::CONNECT,
    )?;
    let service_info = ServiceInfo {
        name: OsString::from(plan.service_name),
        display_name: OsString::from(plan.display_name),
        service_type: ServiceType::OWN_PROCESS,
        start_type: ServiceStartType::AutoStart,
        error_control: ServiceErrorControl::Normal,
        executable_path: plan.executable,
        launch_arguments: plan.launch_arguments.iter().map(OsString::from).collect(),
        dependencies: Vec::new(),
        account_name: Some(OsString::from(r"NT AUTHORITY\LocalService")),
        account_password: None,
    };

    let service = manager.create_service(
        &service_info,
        ServiceAccess::CHANGE_CONFIG
            | ServiceAccess::QUERY_STATUS
            | ServiceAccess::START
            | ServiceAccess::STOP,
    )?;
    service.set_description(SERVICE_DESCRIPTION)?;
    configure_recovery(plan.service_name)?;
    let _ = service.start::<&str>(&[]);
    wait_short();
    Ok(())
}

#[cfg(windows)]
fn uninstall_service_impl() -> Result<()> {
    use windows_service::service::ServiceAccess;
    use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

    let manager = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;
    let service = manager.open_service(
        SERVICE_NAME,
        ServiceAccess::STOP | ServiceAccess::QUERY_STATUS | ServiceAccess::DELETE,
    )?;
    let _ = service.stop();
    wait_short();
    service.delete()?;
    Ok(())
}

#[cfg(windows)]
fn start_service_impl() -> Result<()> {
    use windows_service::service::{ServiceAccess, ServiceState};
    use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

    let manager = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;
    let service = manager.open_service(
        SERVICE_NAME,
        ServiceAccess::START | ServiceAccess::QUERY_STATUS,
    )?;
    if service.query_status()?.current_state == ServiceState::Running {
        return Ok(());
    }
    service.start::<&str>(&[])?;
    Ok(())
}

#[cfg(windows)]
fn stop_service_impl() -> Result<()> {
    use windows_service::service::ServiceAccess;
    use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

    let manager = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;
    let service = manager.open_service(SERVICE_NAME, ServiceAccess::STOP)?;
    service.stop()?;
    Ok(())
}

#[cfg(windows)]
fn configure_recovery(service_name: &str) -> Result<()> {
    use std::process::Command;

    let output = Command::new("sc.exe")
        .args([
            "failure",
            service_name,
            "reset=",
            "86400",
            "actions=",
            "restart/5000/restart/5000/restart/30000",
        ])
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "failed to configure service recovery: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

#[cfg(windows)]
fn wait_short() {
    std::thread::sleep(std::time::Duration::from_millis(500));
}

#[cfg(windows)]
windows_service::define_windows_service!(ffi_service_main, service_main);

#[cfg(windows)]
fn run_service_impl(config: PathBuf) -> Result<()> {
    let _ = SERVICE_CONFIG.set(config);
    windows_service::service_dispatcher::start(SERVICE_NAME, ffi_service_main)
        .map_err(anyhow::Error::from)
        .with_context(|| format!("failed to start service dispatcher for {SERVICE_NAME}"))?;
    Ok(())
}

#[cfg(windows)]
fn service_main(arguments: Vec<std::ffi::OsString>) {
    if let Err(error) = run_service_main(arguments) {
        eprintln!("service failed: {error:?}");
    }
}

#[cfg(windows)]
fn run_service_main(arguments: Vec<std::ffi::OsString>) -> Result<()> {
    use windows_service::service::{ServiceControl, ServiceState};
    use windows_service::service_control_handler::{self, ServiceControlHandlerResult};

    let config = service_config_from_args(arguments)?;
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown_tx = std::sync::Mutex::new(Some(shutdown_tx));

    let status_handle =
        service_control_handler::register(SERVICE_NAME, move |control| match control {
            ServiceControl::Stop | ServiceControl::Shutdown => {
                if let Some(tx) = shutdown_tx.lock().expect("service shutdown mutex").take() {
                    let _ = tx.send(());
                }
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        })?;

    status_handle.set_service_status(service_status(ServiceState::StartPending))?;

    // Load configuration before building the runtime so the worker thread count
    // honors server.thread_count. Report Stopped if early startup fails.
    let startup = (|| -> Result<(crate::config::AppConfig, tokio::runtime::Runtime)> {
        let cfg = crate::config::AppConfig::load(&config)?;
        configure_runtime_environment(&cfg)?;
        crate::logs::init_file_logging(&cfg.logging)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(cfg.server.thread_count.max(1))
            .enable_all()
            .build()?;
        Ok((cfg, runtime))
    })();
    let (cfg, runtime) = match startup {
        Ok(value) => value,
        Err(error) => {
            status_handle.set_service_status(service_status(ServiceState::Stopped))?;
            return Err(error);
        }
    };

    status_handle.set_service_status(service_status(ServiceState::Running))?;
    let result = runtime.block_on(async move {
        tracing::info!("starting ai2npu windows service");
        let openvino = crate::openvino_backend::OpenVinoStatus::detect();
        crate::http::serve_until(cfg, openvino, async {
            let _ = shutdown_rx.await;
        })
        .await
    });

    status_handle.set_service_status(service_status(ServiceState::Stopped))?;
    result
}

#[cfg(windows)]
fn configure_runtime_environment(cfg: &crate::config::AppConfig) -> Result<()> {
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

#[cfg(windows)]
fn service_config_from_args(arguments: Vec<std::ffi::OsString>) -> Result<PathBuf> {
    let mut args = arguments.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--config" {
            return args
                .next()
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("--config requires a path"));
        }
    }
    Ok(SERVICE_CONFIG
        .get()
        .cloned()
        .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData\ai2npu\config.toml")))
}

#[cfg(windows)]
fn service_status(
    state: windows_service::service::ServiceState,
) -> windows_service::service::ServiceStatus {
    windows_service::service::ServiceStatus {
        service_type: windows_service::service::ServiceType::OWN_PROCESS,
        current_state: state,
        controls_accepted: if state == windows_service::service::ServiceState::Running {
            windows_service::service::ServiceControlAccept::STOP
                | windows_service::service::ServiceControlAccept::SHUTDOWN
        } else {
            windows_service::service::ServiceControlAccept::empty()
        },
        exit_code: windows_service::service::ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint: std::time::Duration::from_secs(10),
        process_id: None,
    }
}

#[cfg(not(windows))]
fn install_service_impl(_plan: ServiceInstallPlan) -> Result<()> {
    anyhow::bail!("windows service management is only available on Windows")
}

#[cfg(not(windows))]
fn uninstall_service_impl() -> Result<()> {
    anyhow::bail!("windows service management is only available on Windows")
}

#[cfg(not(windows))]
fn start_service_impl() -> Result<()> {
    anyhow::bail!("windows service management is only available on Windows")
}

#[cfg(not(windows))]
fn stop_service_impl() -> Result<()> {
    anyhow::bail!("windows service management is only available on Windows")
}

#[cfg(not(windows))]
fn run_service_impl(_config: PathBuf) -> Result<()> {
    anyhow::bail!("windows service runtime is only available on Windows")
}
