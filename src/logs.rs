use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::EnvFilter;

use crate::config::LoggingConfig;

#[derive(Debug, Serialize)]
pub struct LogTail {
    pub path: PathBuf,
    pub lines: Vec<String>,
}

pub fn tail_log_file(path: impl AsRef<Path>, requested_lines: usize) -> LogTail {
    let path = path.as_ref();
    let line_count = requested_lines.clamp(1, 1000);
    let mut lines = VecDeque::with_capacity(line_count);

    if let Ok(text) = std::fs::read_to_string(path) {
        for line in text.lines() {
            if lines.len() == line_count {
                lines.pop_front();
            }
            lines.push_back(line.to_string());
        }
    }

    LogTail {
        path: path.to_path_buf(),
        lines: lines.into_iter().collect(),
    }
}

pub fn init_file_logging(config: &LoggingConfig) -> Result<()> {
    fs::create_dir_all(&config.directory)?;
    let path = config.directory.join("ai2npu.log");
    let max_size_bytes = config.max_file_size_mb.saturating_mul(1024 * 1024);
    rotate_log_file(&path, max_size_bytes, config.max_files)?;

    let filter = EnvFilter::try_new(&config.level).unwrap_or_else(|_| EnvFilter::new("info"));
    if let Err(error) = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false)
        .with_writer(LogFileMakeWriter { path })
        .try_init()
    {
        tracing::debug!("file logging subscriber was not installed: {error}");
    }

    Ok(())
}

pub fn rotate_log_file(
    path: impl AsRef<Path>,
    max_size_bytes: u64,
    max_files: usize,
) -> Result<()> {
    let path = path.as_ref();
    if max_files == 0 || max_size_bytes == 0 {
        return Ok(());
    }
    let Ok(metadata) = fs::metadata(path) else {
        return Ok(());
    };
    if metadata.len() <= max_size_bytes {
        return Ok(());
    }

    let oldest = rotated_path(path, max_files);
    if oldest.exists() {
        fs::remove_file(oldest)?;
    }

    for index in (1..max_files).rev() {
        let from = rotated_path(path, index);
        if from.exists() {
            fs::rename(from, rotated_path(path, index + 1))?;
        }
    }

    fs::rename(path, rotated_path(path, 1))?;
    Ok(())
}

fn rotated_path(path: &Path, index: usize) -> PathBuf {
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "ai2npu.log".to_string());
    path.with_file_name(format!("{file_name}.{index}"))
}

struct LogFileMakeWriter {
    path: PathBuf,
}

impl<'a> MakeWriter<'a> for LogFileMakeWriter {
    type Writer = LogFileWriter;

    fn make_writer(&'a self) -> Self::Writer {
        let writer: Box<dyn Write + Send> = match open_log_file(&self.path) {
            Ok(file) => Box::new(file),
            Err(_) => Box::new(io::sink()),
        };
        LogFileWriter { writer }
    }
}

fn open_log_file(path: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

struct LogFileWriter {
    writer: Box<dyn Write + Send>,
}

impl Write for LogFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.writer.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}
