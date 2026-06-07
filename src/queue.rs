use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot, Semaphore};

use crate::audio::AudioOutput;

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceOutput {
    Embeddings(Vec<Vec<f32>>),
    Audio(AudioOutput),
}

type JobFn = Box<dyn FnOnce() -> Result<InferenceOutput> + Send + 'static>;

pub struct InferenceJob {
    operation: JobFn,
}

impl InferenceJob {
    pub fn new(operation: impl FnOnce() -> Result<InferenceOutput> + Send + 'static) -> Self {
        Self {
            operation: Box::new(operation),
        }
    }
}

struct QueuedJob {
    job: InferenceJob,
    started_tx: oneshot::Sender<()>,
    response_tx: oneshot::Sender<Result<InferenceOutput>>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

#[derive(Clone)]
pub struct InferenceQueue {
    tx: mpsc::Sender<QueuedJob>,
    capacity: Arc<Semaphore>,
    max_pending: usize,
}

#[derive(Debug, Error)]
pub enum QueueError {
    #[error("queue is full")]
    Full,
    #[error("queue worker is closed")]
    Closed,
    #[error("queue wait timed out")]
    Timeout,
    #[error(transparent)]
    Job(#[from] anyhow::Error),
}

impl InferenceQueue {
    pub fn new(max_pending: usize) -> Self {
        let (tx, mut rx) = mpsc::channel::<QueuedJob>(max_pending.max(1));
        let capacity = Arc::new(Semaphore::new(max_pending));

        tokio::spawn(async move {
            while let Some(queued) = rx.recv().await {
                let QueuedJob {
                    job,
                    started_tx,
                    response_tx,
                    _permit,
                } = queued;
                drop(_permit);
                if started_tx.send(()).is_err() {
                    continue;
                }
                let result = tokio::task::spawn_blocking(move || (job.operation)())
                    .await
                    .unwrap_or_else(|error| {
                        Err(anyhow::anyhow!("queue worker join failed: {error}"))
                    });
                let _ = response_tx.send(result);
            }
        });

        Self {
            tx,
            capacity,
            max_pending,
        }
    }

    pub fn submit(
        &self,
        job: InferenceJob,
    ) -> impl Future<Output = Result<InferenceOutput, QueueError>> {
        self.submit_inner(job, None)
    }

    pub fn submit_with_timeout(
        &self,
        job: InferenceJob,
        timeout: Duration,
    ) -> impl Future<Output = Result<InferenceOutput, QueueError>> {
        self.submit_inner(job, Some(timeout))
    }

    fn submit_inner(
        &self,
        job: InferenceJob,
        timeout: Option<Duration>,
    ) -> impl Future<Output = Result<InferenceOutput, QueueError>> {
        let tx = self.tx.clone();
        let capacity = Arc::clone(&self.capacity);

        async move {
            let (response_tx, response_rx) = oneshot::channel();
            let (started_tx, started_rx) = oneshot::channel();
            let permit = capacity.try_acquire_owned().map_err(|_| QueueError::Full)?;
            let queued = QueuedJob {
                job,
                started_tx,
                response_tx,
                _permit: permit,
            };

            if let Err(error) = tx.try_send(queued) {
                return match error {
                    mpsc::error::TrySendError::Full(_) => Err(QueueError::Full),
                    mpsc::error::TrySendError::Closed(_) => Err(QueueError::Closed),
                };
            }

            match timeout {
                Some(timeout) => {
                    tokio::select! {
                        started = started_rx => {
                            started.map_err(|_| QueueError::Closed)?;
                        }
                        _ = tokio::time::sleep(timeout) => {
                            return Err(QueueError::Timeout);
                        }
                    }
                }
                None => {
                    started_rx.await.map_err(|_| QueueError::Closed)?;
                }
            }

            let result = response_rx.await.map_err(|_| QueueError::Closed)?;
            result.map_err(QueueError::Job)
        }
    }

    pub fn pending_len(&self) -> usize {
        self.max_pending
            .saturating_sub(self.capacity.available_permits())
    }
}
