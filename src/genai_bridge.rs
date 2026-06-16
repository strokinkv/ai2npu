use std::ffi::{c_char, c_float, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use libloading::{Library, Symbol};

use crate::audio::{AudioEndpoint, AudioOutput};
use crate::inference::AudioInferenceOptions;

type WhisperCreate =
    unsafe extern "C" fn(*const c_char, *const c_char, *mut *mut c_void, *mut *mut c_char) -> i32;
type WhisperTranscribe = unsafe extern "C" fn(
    *mut c_void,
    *const c_float,
    usize,
    *const c_char,
    *const c_char,
    *const c_char,
    c_float,
    bool,
    *mut *mut c_char,
    *mut *mut c_char,
) -> i32;
type WhisperFree = unsafe extern "C" fn(*mut c_void);
type FreeString = unsafe extern "C" fn(*mut c_char);

pub struct GenAiWhisperBridge {
    _library: Library,
    create: WhisperCreate,
    transcribe: WhisperTranscribe,
    free: WhisperFree,
    free_string: FreeString,
}

impl GenAiWhisperBridge {
    pub fn load_default() -> Result<Arc<Self>> {
        Self::load(default_bridge_path()?)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let path = path.as_ref();
        if !path.is_file() {
            bail!(
                "native GenAI bridge DLL not found at {}; set AI2NPU_GENAI_BRIDGE_DLL or build native/ai2npu_genai_bridge",
                path.display()
            );
        }

        let library = unsafe { Library::new(path) }
            .with_context(|| format!("failed to load native GenAI bridge at {}", path.display()))?;
        let create = load_symbol::<WhisperCreate>(&library, b"ai2npu_whisper_create\0")?;
        let transcribe =
            load_symbol::<WhisperTranscribe>(&library, b"ai2npu_whisper_transcribe\0")?;
        let free = load_symbol::<WhisperFree>(&library, b"ai2npu_whisper_free\0")?;
        let free_string = load_symbol::<FreeString>(&library, b"ai2npu_whisper_free_string\0")?;

        Ok(Arc::new(Self {
            _library: library,
            create,
            transcribe,
            free,
            free_string,
        }))
    }

    pub fn create_session(
        self: &Arc<Self>,
        model_dir: &Path,
        device: &str,
    ) -> Result<GenAiWhisperSession> {
        let model_dir = path_to_cstring(model_dir)?;
        let device = CString::new(device)?;
        let mut handle = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status =
            unsafe { (self.create)(model_dir.as_ptr(), device.as_ptr(), &mut handle, &mut error) };
        if status != 0 || handle.is_null() {
            bail!(
                "native GenAI bridge failed to create Whisper pipeline: {}",
                unsafe { self.take_string(error) }
            );
        }

        Ok(GenAiWhisperSession {
            bridge: Arc::clone(self),
            handle,
        })
    }

    unsafe fn take_string(&self, ptr: *mut c_char) -> String {
        if ptr.is_null() {
            return "unknown error".to_string();
        }
        let text = CStr::from_ptr(ptr).to_string_lossy().into_owned();
        (self.free_string)(ptr);
        text
    }
}

pub struct GenAiWhisperSession {
    bridge: Arc<GenAiWhisperBridge>,
    handle: *mut c_void,
}

unsafe impl Send for GenAiWhisperSession {}

impl GenAiWhisperSession {
    pub fn transcribe(
        &self,
        samples: &[f32],
        options: &AudioInferenceOptions,
    ) -> Result<AudioOutput> {
        let task = CString::new(match options.endpoint {
            AudioEndpoint::Transcriptions => "transcribe",
            AudioEndpoint::Translations => "translate",
        })?;
        let language = optional_cstring(options.language.as_deref())?;
        let prompt = optional_cstring(options.prompt.as_deref())?;
        // Negative temperature signals "unset" to the native bridge, which then
        // keeps the model's default generation temperature.
        let temperature = options.temperature.unwrap_or(-1.0);
        let mut json_out = ptr::null_mut();
        let mut error = ptr::null_mut();

        let status = unsafe {
            (self.bridge.transcribe)(
                self.handle,
                samples.as_ptr(),
                samples.len(),
                task.as_ptr(),
                language
                    .as_ref()
                    .map_or(ptr::null(), |value| value.as_ptr()),
                prompt.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                temperature,
                options.return_timestamps,
                &mut json_out,
                &mut error,
            )
        };
        if status != 0 {
            bail!("native GenAI bridge transcription failed: {}", unsafe {
                self.bridge.take_string(error)
            });
        }

        let output = unsafe { self.bridge.take_string(json_out) };
        serde_json::from_str(&output).context("failed to parse native GenAI bridge output")
    }
}

impl Drop for GenAiWhisperSession {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.bridge.free)(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

fn default_bridge_path() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("AI2NPU_GENAI_BRIDGE_DLL") {
        return Ok(PathBuf::from(path));
    }
    Ok(std::env::current_exe()?
        .parent()
        .context("failed to resolve executable directory")?
        .join("ai2npu_genai_bridge.dll"))
}

fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    let symbol: Symbol<'_, T> = unsafe { library.get(name) }
        .with_context(|| format!("native GenAI bridge missing symbol {}", symbol_name(name)))?;
    Ok(*symbol)
}

fn symbol_name(name: &[u8]) -> String {
    String::from_utf8_lossy(name.strip_suffix(&[0]).unwrap_or(name)).into_owned()
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes()).context("path contains nul byte")
}

fn optional_cstring(value: Option<&str>) -> Result<Option<CString>> {
    value
        .map(CString::new)
        .transpose()
        .context("string contains nul byte")
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    static FREE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    unsafe extern "C" fn create_stub(
        _model_dir: *const c_char,
        _device: *const c_char,
        _handle: *mut *mut c_void,
        _error: *mut *mut c_char,
    ) -> i32 {
        0
    }

    unsafe extern "C" fn transcribe_stub(
        _handle: *mut c_void,
        _samples: *const c_float,
        _sample_count: usize,
        _task: *const c_char,
        _language: *const c_char,
        _prompt: *const c_char,
        _temperature: c_float,
        _return_timestamps: bool,
        _json_out: *mut *mut c_char,
        _error: *mut *mut c_char,
    ) -> i32 {
        0
    }

    unsafe extern "C" fn free_stub(_handle: *mut c_void) {
        FREE_CALLS.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn free_string_stub(_value: *mut c_char) {}

    fn bridge_for_drop_test() -> Arc<GenAiWhisperBridge> {
        Arc::new(GenAiWhisperBridge {
            _library: unsafe { Library::new("kernel32.dll") }.unwrap(),
            create: create_stub,
            transcribe: transcribe_stub,
            free: free_stub,
            free_string: free_string_stub,
        })
    }

    #[test]
    fn drop_frees_native_whisper_handle_by_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("AI2NPU_FREE_NATIVE_WHISPER_ON_DROP");
        FREE_CALLS.store(0, Ordering::SeqCst);

        drop(GenAiWhisperSession {
            bridge: bridge_for_drop_test(),
            handle: ptr::dangling_mut::<c_void>(),
        });

        assert_eq!(FREE_CALLS.load(Ordering::SeqCst), 1);
    }
}
