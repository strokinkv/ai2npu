use std::sync::{Arc, Mutex};
use std::time::Duration;

use ai2npu::queue::{InferenceJob, InferenceOutput, InferenceQueue, QueueError};

#[tokio::test]
async fn completes_jobs_in_fifo_order() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let queue = InferenceQueue::new(10);

    let first_order = Arc::clone(&order);
    let first = queue.submit(InferenceJob::new(move || {
        first_order.lock().unwrap().push(1);
        Ok(InferenceOutput::Embeddings(vec![vec![1.0]]))
    }));

    let second_order = Arc::clone(&order);
    let second = queue.submit(InferenceJob::new(move || {
        second_order.lock().unwrap().push(2);
        Ok(InferenceOutput::Embeddings(vec![vec![2.0]]))
    }));

    let (first, second) = tokio::join!(first, second);

    assert_eq!(first.unwrap(), InferenceOutput::Embeddings(vec![vec![1.0]]));
    assert_eq!(
        second.unwrap(),
        InferenceOutput::Embeddings(vec![vec![2.0]])
    );
    assert_eq!(*order.lock().unwrap(), vec![1, 2]);
}

#[tokio::test]
async fn returns_queue_full_when_pending_capacity_is_exceeded() {
    let queue = InferenceQueue::new(0);

    let result = queue
        .submit(InferenceJob::new(|| {
            Ok(InferenceOutput::Embeddings(vec![vec![1.0]]))
        }))
        .await;

    assert!(matches!(result, Err(QueueError::Full)));
}

#[tokio::test]
async fn returns_queue_timeout_when_waiting_too_long() {
    let queue = InferenceQueue::new(1);
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();

    let first_queue = queue.clone();
    let first = tokio::spawn(async move {
        first_queue
            .submit(InferenceJob::new(move || {
                release_rx.recv().unwrap();
                Ok(InferenceOutput::Embeddings(vec![vec![1.0]]))
            }))
            .await
    });
    tokio::task::yield_now().await;

    let second = queue
        .submit_with_timeout(
            InferenceJob::new(|| Ok(InferenceOutput::Embeddings(vec![vec![2.0]]))),
            Duration::from_millis(10),
        )
        .await;

    release_tx.send(()).unwrap();
    let _ = first.await.unwrap().unwrap();

    assert!(matches!(second, Err(QueueError::Timeout)));
}
