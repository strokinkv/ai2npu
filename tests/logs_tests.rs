use ai2npu::logs::rotate_log_file;

#[test]
fn rotates_log_file_by_size_and_retains_max_files() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("ai2npu.log");
    std::fs::write(&log_path, "current log over limit").unwrap();
    std::fs::write(dir.path().join("ai2npu.log.1"), "old one").unwrap();
    std::fs::write(dir.path().join("ai2npu.log.2"), "old two").unwrap();

    rotate_log_file(&log_path, 4, 2).unwrap();

    assert!(!log_path.exists());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("ai2npu.log.1")).unwrap(),
        "current log over limit"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("ai2npu.log.2")).unwrap(),
        "old one"
    );
    assert!(!dir.path().join("ai2npu.log.3").exists());
}
