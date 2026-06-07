use std::{
    collections::HashSet,
    fmt, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

const DOWNLOAD_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const DOWNLOAD_RETRIES: usize = 3;
const DOWNLOAD_RETRY_DELAY: Duration = Duration::from_secs(2);

#[derive(Debug)]
struct RetryableHttpStatus {
    status: reqwest::StatusCode,
    url: String,
}

impl fmt::Display for RetryableHttpStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "download failed with transient HTTP status {} for {}",
            self.status, self.url
        )
    }
}

impl std::error::Error for RetryableHttpStatus {}

const WHISPER_REPO_URL: &str = "https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int8-ov";
const BGE_REPO_URL: &str = "https://huggingface.co/strokinkv/bge-m3-int8-ov";
const WHISPER_REVISION: &str = "main";

#[derive(Clone, Copy)]
struct ManifestEntry {
    path: &'static str,
    size: u64,
    sha256: &'static str,
}

const WHISPER_MANIFEST: &[ManifestEntry] = &[
    ManifestEntry {
        path: ".gitattributes",
        size: 1519,
        sha256: "11AD7EFA24975EE4B0C3C3A38ED18737F0658A5F75A0A96787B576A78A023361",
    },
    ManifestEntry {
        path: "added_tokens.json",
        size: 34648,
        sha256: "3C51F66C4C21F9E126970078F11AE77A78C74AEE8DF606EE9DABA86E467108E0",
    },
    ManifestEntry {
        path: "config.json",
        size: 1192,
        sha256: "BFD92C097547AB12CB42ABAE8008BE5A59A91FDC5AB39ACCE24489EB8A3E8A86",
    },
    ManifestEntry {
        path: "generation_config.json",
        size: 3767,
        sha256: "4617FCCA458AF3B91A103143AAAC919C1AB6680B552D7ABD10811B7248BD77B4",
    },
    ManifestEntry {
        path: "merges.txt",
        size: 493869,
        sha256: "2DF2990A395E35E8DFBC7511E08C12D56018D8D04691E0133E5D63B21E154DC6",
    },
    ManifestEntry {
        path: "normalizer.json",
        size: 52666,
        sha256: "BF1C507DC8724CA9CF9903640DACFB69DAE2F00EDEE4F21CEBA106A7392F26DD",
    },
    ManifestEntry {
        path: "openvino_config.json",
        size: 622,
        sha256: "9DA38E88CEC069AD54699D627F1C59D36C36A3DFB54B5E1BB5CFDF5832EFAF03",
    },
    ManifestEntry {
        path: "openvino_decoder_model.bin",
        size: 172534710,
        sha256: "C064991CBAFC4381567D29972B7013DC24026DE9C326D03EB1E6E4FC44AA959F",
    },
    ManifestEntry {
        path: "openvino_decoder_model.xml",
        size: 391642,
        sha256: "AEB09FAFBF1C0CBF84BAF30F46763436005822FAF3B365359B8DE0AA04F03047",
    },
    ManifestEntry {
        path: "openvino_detokenizer.bin",
        size: 736198,
        sha256: "F2B3C47825A1089525FF65C0C8E49271E1DEE69A401A04FC827AC2DE5B7766E4",
    },
    ManifestEntry {
        path: "openvino_detokenizer.xml",
        size: 9779,
        sha256: "6E106A14F14B0771B46B7948A99B1D819FF93B2455B7DA8F47761AB9DBA9DC56",
    },
    ManifestEntry {
        path: "openvino_encoder_model.bin",
        size: 645332592,
        sha256: "0590A8F35F96D57801C55990028D917821AC721026E34B7F3F59D7561FC908E6",
    },
    ManifestEntry {
        path: "openvino_encoder_model.xml",
        size: 1518660,
        sha256: "60713D4ED3A8AC8EE020E11C4737EC276D14CABC6A082537BDDF2C00BA6CE070",
    },
    ManifestEntry {
        path: "openvino_tokenizer.bin",
        size: 1898973,
        sha256: "ADFA3D9A2920D0F314121270A960AB331EC0F05838544BB8ECAAA422282A6FD4",
    },
    ManifestEntry {
        path: "openvino_tokenizer.xml",
        size: 27091,
        sha256: "CBA304E7BAD54773B9D2CBCCFBC8501117ECF2E3C0F4F5331742A0A3C9FEED93",
    },
    ManifestEntry {
        path: "preprocessor_config.json",
        size: 357,
        sha256: "654CF18D3E163B948CEAF9766DA56CE0B52DE265D58673CF61C9376F126BD499",
    },
    ManifestEntry {
        path: "README.md",
        size: 5130,
        sha256: "5E65CE8606306CB7E71E13CBA40A89E1309A95070E41D5591EEF0195238EB5FB",
    },
    ManifestEntry {
        path: "special_tokens_map.json",
        size: 2186,
        sha256: "BAEA4EA09372EB4FCA86B4E4346139FD73CB807D5087E9DE0948E971739C3E74",
    },
    ManifestEntry {
        path: "tokenizer_config.json",
        size: 282873,
        sha256: "3C75940DFCE3A294FCA7041A5FAFF011677F1B68FA85E47511BB8CF6DCCADED6",
    },
    ManifestEntry {
        path: "tokenizer.json",
        size: 3930645,
        sha256: "5C1BF30C9E716E1477BEDEF846B01BE0013DAECB89E9E3EF7AB89B23C178DF1B",
    },
    ManifestEntry {
        path: "vocab.json",
        size: 835528,
        sha256: "6788C80B082E9B0D1393147D3A3E62BA19285AC0C82ACE8E5EF00F37EAD58971",
    },
];

const BGE_MANIFEST: &[ManifestEntry] = &[
    ManifestEntry {
        path: ".gitattributes",
        size: 1570,
        sha256: "34448B82C17D60FEC9B65B1F093C115DDBAADC04BEB1B0140B6BFED2E012A930",
    },
    ManifestEntry {
        path: "config.json",
        size: 658,
        sha256: "70DAE5884CED999AF00244F776AC9EAA71538D68497D3D6A6091E0318CD32905",
    },
    ManifestEntry {
        path: "model.bin",
        size: 569182363,
        sha256: "316B1C85B754ADAEE06CAD60B7E6794F533D5AB7A1353FE310CF854692BDAE1F",
    },
    ManifestEntry {
        path: "model.xml",
        size: 1231800,
        sha256: "AEC433B8E359225B62AFFBE3953470C556687006D1E3D4C85B8AC37138724D59",
    },
    ManifestEntry {
        path: "README.md",
        size: 3520,
        sha256: "FE648FD4636C7473F7B18499615423646C35D8197D14244416881617F294DACD",
    },
    ManifestEntry {
        path: "README.ru.md",
        size: 3955,
        sha256: "80688BF3D117E82F9162A734AFFB82F414F3A56424606655D705F291E7F1AF7D",
    },
    ManifestEntry {
        path: "openvino_config.json",
        size: 418,
        sha256: "34A3A81FDA54346262EFC3E5AF6D4F178C5EBFAC9A7D23706FBA3B9300CB29E0",
    },
    ManifestEntry {
        path: "openvino_model.bin",
        size: 569173583,
        sha256: "80DD5181E8432B761AC4CAB673CEE52F287B27CCC3E3A358FD8E7CF14F811C05",
    },
    ManifestEntry {
        path: "openvino_model.xml",
        size: 1169180,
        sha256: "F8E9C263EE65359BF10A0D769770CFE803A6ADCAD93E61864CA456663FE4109E",
    },
    ManifestEntry {
        path: "sentencepiece.bpe.model",
        size: 5069051,
        sha256: "CFC8146ABE2A0488E9E2A0C56DE7952F7C11AB059ECA145A0A727AFCE0DB2865",
    },
    ManifestEntry {
        path: "special_tokens_map.json",
        size: 964,
        sha256: "8C785ABEBEA9AE3257B61681B4E6FD8365CEAFDE980C21970D001E834CF10835",
    },
    ManifestEntry {
        path: "tokenizer_config.json",
        size: 1203,
        sha256: "B87C8703482B0300D3DA30E201519AA641F6A450F5EB5BF1E624AFBF70C74D80",
    },
    ManifestEntry {
        path: "tokenizer.json",
        size: 17082799,
        sha256: "249DF0778F236F6ECE390DE0DE746838EF25B9D6954B68C2EE71249E0A9D8FD4",
    },
];

pub fn install_model(model: &str, model_dir: &Path) -> Result<()> {
    match model {
        "openai/whisper-large-v3-turbo" | "whisper-large-v3-turbo-int8-ov" => {
            install_manifest_model(
                model_dir,
                WHISPER_REPO_URL,
                WHISPER_REVISION,
                WHISPER_MANIFEST,
            )
        }
        "BAAI/bge-m3" | "strokinkv/bge-m3-int8-ov" | "bge-m3-int8-ov" => {
            install_manifest_model(model_dir, BGE_REPO_URL, WHISPER_REVISION, BGE_MANIFEST)
        }
        other => bail!("unsupported install-model target: {other}"),
    }
}

fn install_manifest_model(
    model_dir: &Path,
    repo_url: &str,
    revision: &str,
    manifest: &[ManifestEntry],
) -> Result<()> {
    fs::create_dir_all(model_dir)
        .with_context(|| format!("failed to create model directory {}", model_dir.display()))?;
    let model_dir = model_dir.canonicalize().with_context(|| {
        format!(
            "failed to resolve model directory after creation: {}",
            model_dir.display()
        )
    })?;
    let temp_dir = model_dir.join(".ai2npu-download");
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)
            .with_context(|| format!("failed to remove {}", temp_dir.display()))?;
    }
    fs::create_dir_all(&temp_dir)
        .with_context(|| format!("failed to create {}", temp_dir.display()))?;

    let result = install_manifest_model_inner(&model_dir, &temp_dir, repo_url, revision, manifest);
    let cleanup = fs::remove_dir_all(&temp_dir)
        .with_context(|| format!("failed to remove {}", temp_dir.display()));
    result.and(cleanup)
}

fn install_manifest_model_inner(
    model_dir: &Path,
    temp_dir: &Path,
    repo_url: &str,
    revision: &str,
    manifest: &[ManifestEntry],
) -> Result<()> {
    for entry in manifest {
        let target = model_dir.join(entry.path);
        if expected_file_matches(&target, entry)? {
            println!("OK {}", entry.path);
            continue;
        }

        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        let temp_file = temp_dir.join(format!("{}.tmp", entry.path.replace(['/', '\\'], "_")));
        let url = format!("{}/resolve/{}/{}", repo_url, revision, hf_path(entry.path));
        println!("Downloading {}", entry.path);
        download_file(&url, &temp_file)
            .with_context(|| format!("failed to download {}", entry.path))?;

        if !expected_file_matches(&temp_file, entry)? {
            bail!("downloaded file failed verification: {}", entry.path);
        }

        fs::rename(&temp_file, &target)
            .or_else(|_| {
                fs::copy(&temp_file, &target)?;
                fs::remove_file(&temp_file)
            })
            .with_context(|| {
                format!(
                    "failed to move {} to {}",
                    temp_file.display(),
                    target.display()
                )
            })?;
    }

    remove_stale_files(model_dir, temp_dir, manifest)?;
    let git_dir = model_dir.join(".git");
    if git_dir.exists() {
        fs::remove_dir_all(&git_dir)
            .with_context(|| format!("failed to remove {}", git_dir.display()))?;
    }
    Ok(())
}

fn download_file(url: &str, destination: &Path) -> Result<()> {
    let url = reqwest::Url::parse(url).with_context(|| format!("invalid download URL: {url}"))?;
    if url.scheme() != "https" {
        bail!("model downloads require HTTPS URLs: {url}");
    }

    let client = reqwest::blocking::Client::builder()
        .connect_timeout(DOWNLOAD_CONNECT_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(10))
        .https_only(true)
        .use_rustls_tls()
        .build()
        .context("failed to initialize HTTPS client for model download")?;

    let mut last_error = None;
    for attempt in 0..=DOWNLOAD_RETRIES {
        match download_file_once(&client, url.clone(), destination) {
            Ok(()) => return Ok(()),
            Err(error) if attempt < DOWNLOAD_RETRIES && is_retryable_download_error(&error) => {
                last_error = Some(error);
                thread::sleep(DOWNLOAD_RETRY_DELAY);
            }
            Err(error) => return Err(error),
        }
    }

    Err(last_error.expect("download retry loop must record an error"))
}

fn download_file_once(
    client: &reqwest::blocking::Client,
    url: reqwest::Url,
    destination: &Path,
) -> Result<()> {
    let mut response = client
        .get(url.clone())
        .send()
        .with_context(|| format!("failed to request {url}"))?;

    let status = response.status();
    if !status.is_success() {
        if is_retryable_http_status(status) {
            return Err(RetryableHttpStatus {
                status,
                url: url.to_string(),
            }
            .into());
        }
        bail!("download failed with HTTP status {status} for {url}");
    }

    let mut output = fs::File::create(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    std::io::copy(&mut response, &mut output)
        .with_context(|| format!("failed to write {}", destination.display()))?;
    output
        .flush()
        .with_context(|| format!("failed to flush {}", destination.display()))?;
    Ok(())
}

fn is_retryable_download_error(error: &anyhow::Error) -> bool {
    if error
        .chain()
        .any(|cause| cause.downcast_ref::<RetryableHttpStatus>().is_some())
    {
        return true;
    }

    if error.chain().any(|cause| {
        cause
            .downcast_ref::<reqwest::Error>()
            .is_some_and(|reqwest_error| {
                reqwest_error.is_timeout()
                    || reqwest_error.is_connect()
                    || reqwest_error.is_request()
                    || reqwest_error.is_body()
            })
    }) {
        return true;
    }

    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io_error| {
                matches!(
                    io_error.kind(),
                    std::io::ErrorKind::ConnectionAborted
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::Interrupted
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::UnexpectedEof
                        | std::io::ErrorKind::WouldBlock
                )
            })
    })
}

fn is_retryable_http_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::INTERNAL_SERVER_ERROR
        || status == reqwest::StatusCode::BAD_GATEWAY
        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
        || status == reqwest::StatusCode::GATEWAY_TIMEOUT
}

fn expected_file_matches(path: &Path, entry: &ManifestEntry) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;
    if metadata.len() != entry.size {
        return Ok(false);
    }
    let actual = sha256_file(path)?;
    Ok(actual.eq_ignore_ascii_case(entry.sha256))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode_upper(hasher.finalize()))
}

fn remove_stale_files(model_dir: &Path, temp_dir: &Path, manifest: &[ManifestEntry]) -> Result<()> {
    let expected: HashSet<String> = manifest
        .iter()
        .map(|entry| normalize_relative(entry.path))
        .collect();
    let mut files = Vec::new();
    collect_files(model_dir, Some(temp_dir), &mut files)?;

    for path in files {
        if path.starts_with(temp_dir) {
            continue;
        }
        let relative = path
            .strip_prefix(model_dir)
            .with_context(|| format!("failed to relativize {}", path.display()))?;
        let normalized = normalize_path(relative);
        if !expected.contains(&normalized) {
            println!("Removing stale file {normalized}");
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove stale file {}", path.display()))?;
        }
    }
    Ok(())
}

fn collect_files(dir: &Path, skip_dir: Option<&Path>, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if skip_dir.is_some_and(|skip| path == skip) {
            continue;
        }
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files(&path, skip_dir, out)?;
        } else if file_type.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn normalize_relative(path: &str) -> String {
    path.replace('\\', "/").to_ascii_lowercase()
}

fn normalize_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
        .to_ascii_lowercase()
}

fn hf_path(path: &str) -> String {
    path.split(['/', '\\'])
        .map(percent_encode_path_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn percent_encode_path_segment(segment: &str) -> String {
    let mut encoded = String::new();
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_file_rejects_non_https_urls_without_creating_destination() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let destination = temp_dir.path().join("model.bin");

        let error = download_file("http://127.0.0.1:9/model.bin", &destination)
            .expect_err("non-HTTPS downloads must be rejected");

        assert!(
            error.to_string().contains("HTTPS"),
            "unexpected error: {error:#}"
        );
        assert!(!destination.exists());
    }
}
