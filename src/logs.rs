use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

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
        .with_writer(LogFileMakeWriter::new(
            path,
            max_size_bytes,
            config.max_files,
        )?)
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

    rotate_log_file_unchecked(path, max_files)
}

fn rotate_log_file_unchecked(path: &Path, max_files: usize) -> Result<()> {
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
    state: Arc<Mutex<LogFileState>>,
}

impl LogFileMakeWriter {
    fn new(path: PathBuf, max_size_bytes: u64, max_files: usize) -> io::Result<Self> {
        Ok(Self {
            state: Arc::new(Mutex::new(LogFileState::new(
                path,
                max_size_bytes,
                max_files,
            )?)),
        })
    }
}

struct LogFileState {
    path: PathBuf,
    max_size_bytes: u64,
    max_files: usize,
    file: Option<File>,
    current_size: u64,
}

impl LogFileState {
    fn new(path: PathBuf, max_size_bytes: u64, max_files: usize) -> io::Result<Self> {
        let file = open_log_file(&path)?;
        let current_size = file.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        Ok(Self {
            path,
            max_size_bytes,
            max_files,
            file: Some(file),
            current_size,
        })
    }

    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.rotate_before_write(buf.len() as u64)?;
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("log file is not open"))?;
        file.write_all(buf)?;
        self.current_size = self.current_size.saturating_add(buf.len() as u64);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("log file is not open"))?;
        file.flush()?;
        Ok(())
    }

    fn rotate_before_write(&mut self, next_len: u64) -> io::Result<()> {
        if self.max_files == 0
            || self.max_size_bytes == 0
            || self.current_size.saturating_add(next_len) <= self.max_size_bytes
            || self.current_size == 0
        {
            return Ok(());
        }

        if let Some(file) = self.file.as_mut() {
            file.flush()?;
        }
        self.file.take();
        if let Err(error) = rotate_log_file_unchecked(&self.path, self.max_files) {
            let _ = self.reopen_active_file();
            return Err(io::Error::other(error.to_string()));
        }
        self.reopen_active_file()
    }

    fn reopen_active_file(&mut self) -> io::Result<()> {
        let file = open_log_file(&self.path)?;
        self.current_size = file.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        self.file = Some(file);
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for LogFileMakeWriter {
    type Writer = LogFileWriter<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        match self.state.lock() {
            Ok(state) => LogFileWriter::new(state),
            Err(_) => LogFileWriter::poisoned(),
        }
    }
}

fn open_log_file(path: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

struct LogFileWriter<'a> {
    state: Option<MutexGuard<'a, LogFileState>>,
}

impl<'a> LogFileWriter<'a> {
    fn new(state: MutexGuard<'a, LogFileState>) -> Self {
        Self { state: Some(state) }
    }

    fn poisoned() -> Self {
        Self { state: None }
    }

    fn state(&mut self) -> io::Result<&mut LogFileState> {
        self.state
            .as_deref_mut()
            .ok_or_else(|| io::Error::other("log writer mutex poisoned"))
    }
}

impl Write for LogFileWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.state()?.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.state()?.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Arc, Barrier};

    use super::*;

    #[test]
    fn concurrent_log_writes_are_retained_across_rotated_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ai2npu.log");
        std::fs::write(&path, vec![b'x'; 512]).unwrap();
        let writer = Arc::new(LogFileMakeWriter::new(path.clone(), 256, 32).unwrap());
        let barrier = Arc::new(Barrier::new(32));
        let mut handles = Vec::new();

        for index in 0..32 {
            let writer = Arc::clone(&writer);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let mut log = writer.make_writer();
                log.write_all(format!("concurrent-log-line-{index:02}\n").as_bytes())
                    .unwrap();
                log.flush().unwrap();
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let mut lines = HashSet::new();
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let entry = entry.unwrap();
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with("ai2npu.log")
            {
                let text = std::fs::read_to_string(entry.path()).unwrap();
                lines.extend(text.lines().map(str::to_owned));
            }
        }

        for index in 0..32 {
            assert!(
                lines.contains(&format!("concurrent-log-line-{index:02}")),
                "missing concurrent log line {index}"
            );
        }
    }

    #[test]
    fn multi_write_log_events_are_flushed_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ai2npu.log");
        let writer = Arc::new(LogFileMakeWriter::new(path.clone(), 1024, 2).unwrap());

        let mut first = writer.make_writer();
        first.write_all(b"first-").unwrap();

        let second_writer = Arc::clone(&writer);
        let second = std::thread::spawn(move || {
            let mut second = second_writer.make_writer();
            second.write_all(b"second-event\n").unwrap();
            second.flush().unwrap();
        });

        first.write_all(b"event\n").unwrap();
        first.flush().unwrap();
        drop(first);
        second.join().unwrap();

        let text = std::fs::read_to_string(path).unwrap();
        assert!(text.lines().any(|line| line == "first-event"));
        assert!(text.lines().any(|line| line == "second-event"));
    }

    #[test]
    fn rotates_before_write_that_would_cross_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ai2npu.log");
        std::fs::write(&path, "abc").unwrap();
        let writer = LogFileMakeWriter::new(path.clone(), 5, 2).unwrap();

        let mut log = writer.make_writer();
        log.write_all(b"def\n").unwrap();
        log.flush().unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "def\n");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("ai2npu.log.1")).unwrap(),
            "abc"
        );
    }
}
