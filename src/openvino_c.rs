use anyhow::{bail, Result};
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::ptr;
use std::slice;

#[repr(C)]
struct ov_core_t {
    _private: [u8; 0],
}

#[repr(C)]
struct ov_model_t {
    _private: [u8; 0],
}

#[repr(C)]
struct ov_compiled_model_t {
    _private: [u8; 0],
}

#[repr(C)]
struct ov_infer_request_t {
    _private: [u8; 0],
}

#[repr(C)]
struct ov_tensor_t {
    _private: [u8; 0],
}

#[repr(C)]
struct ov_output_const_port_t {
    _private: [u8; 0],
}

#[repr(C)]
struct ov_shape_t {
    rank: i64,
    dims: *mut i64,
}

#[repr(C)]
struct ov_available_devices_t {
    devices: *mut *mut c_char,
    size: usize,
}

const OV_ELEMENT_TYPE_F32: i32 = 4;
const OV_ELEMENT_TYPE_I64: i32 = 10;

#[link(name = "openvino_c")]
extern "C" {
    fn ov_core_create(core: *mut *mut ov_core_t) -> i32;
    fn ov_core_free(core: *mut ov_core_t);
    fn ov_core_read_model(
        core: *const ov_core_t,
        model_path: *const c_char,
        bin_path: *const c_char,
        model: *mut *mut ov_model_t,
    ) -> i32;
    fn ov_core_compile_model(
        core: *const ov_core_t,
        model: *const ov_model_t,
        device_name: *const c_char,
        property_args_size: usize,
        compiled_model: *mut *mut ov_compiled_model_t,
    ) -> i32;
    fn ov_core_get_available_devices(
        core: *const ov_core_t,
        devices: *mut ov_available_devices_t,
    ) -> i32;
    fn ov_available_devices_free(devices: *mut ov_available_devices_t);
    fn ov_model_free(model: *mut ov_model_t);
    fn ov_compiled_model_outputs_size(
        compiled_model: *const ov_compiled_model_t,
        size: *mut usize,
    ) -> i32;
    fn ov_compiled_model_output_by_index(
        compiled_model: *const ov_compiled_model_t,
        index: usize,
        output_port: *mut *mut ov_output_const_port_t,
    ) -> i32;
    fn ov_compiled_model_free(compiled_model: *mut ov_compiled_model_t);
    fn ov_compiled_model_create_infer_request(
        compiled_model: *const ov_compiled_model_t,
        infer_request: *mut *mut ov_infer_request_t,
    ) -> i32;
    fn ov_infer_request_set_tensor(
        infer_request: *mut ov_infer_request_t,
        tensor_name: *const c_char,
        tensor: *const ov_tensor_t,
    ) -> i32;
    fn ov_infer_request_get_tensor(
        infer_request: *const ov_infer_request_t,
        tensor_name: *const c_char,
        tensor: *mut *mut ov_tensor_t,
    ) -> i32;
    fn ov_infer_request_infer(infer_request: *mut ov_infer_request_t) -> i32;
    fn ov_infer_request_free(infer_request: *mut ov_infer_request_t);
    fn ov_tensor_create_from_host_ptr(
        element_type: i32,
        shape: ov_shape_t,
        host_ptr: *mut c_void,
        tensor: *mut *mut ov_tensor_t,
    ) -> i32;
    fn ov_tensor_get_element_type(tensor: *const ov_tensor_t, element_type: *mut i32) -> i32;
    fn ov_tensor_get_size(tensor: *const ov_tensor_t, elements_size: *mut usize) -> i32;
    fn ov_tensor_data(tensor: *const ov_tensor_t, data: *mut *mut c_void) -> i32;
    fn ov_tensor_free(tensor: *mut ov_tensor_t);
    fn ov_port_get_any_name(
        port: *const ov_output_const_port_t,
        tensor_name: *mut *mut c_char,
    ) -> i32;
    fn ov_output_const_port_free(port: *mut ov_output_const_port_t);
    fn ov_free(content: *const c_char);
}

pub struct OpenVinoRuntime {
    core: *mut ov_core_t,
}

impl OpenVinoRuntime {
    pub fn new() -> Result<Self> {
        let mut core = ptr::null_mut();
        let status = unsafe { ov_core_create(&mut core) };
        if status != 0 || core.is_null() {
            bail!("ov_core_create failed with status {status}");
        }

        Ok(Self { core })
    }

    pub fn available_devices(&self) -> Result<Vec<String>> {
        let mut raw = ov_available_devices_t {
            devices: ptr::null_mut(),
            size: 0,
        };
        let status = unsafe { ov_core_get_available_devices(self.core, &mut raw) };
        if status != 0 {
            bail!("ov_core_get_available_devices failed with status {status}");
        }
        if raw.size > 0 && raw.devices.is_null() {
            bail!("ov_core_get_available_devices returned a null devices array");
        }
        if raw.size > 1024 {
            unsafe { ov_available_devices_free(&mut raw) };
            bail!(
                "ov_core_get_available_devices returned an implausible device count: {}",
                raw.size
            );
        }

        let mut devices = Vec::with_capacity(raw.size);
        for index in 0..raw.size {
            let device = unsafe { *raw.devices.add(index) };
            if !device.is_null() {
                devices.push(
                    unsafe { CStr::from_ptr(device) }
                        .to_string_lossy()
                        .to_string(),
                );
            }
        }

        unsafe { ov_available_devices_free(&mut raw) };
        Ok(devices)
    }

    pub fn read_model(&self, xml_path: impl AsRef<Path>) -> Result<OpenVinoModel> {
        let xml_path = cstring_path(xml_path.as_ref())?;
        let mut model = ptr::null_mut();
        let status =
            unsafe { ov_core_read_model(self.core, xml_path.as_ptr(), ptr::null(), &mut model) };
        if status != 0 || model.is_null() {
            bail!("ov_core_read_model failed with status {status}");
        }

        Ok(OpenVinoModel { raw: model })
    }

    pub fn compile_model(&self, model: &OpenVinoModel, device: &str) -> Result<CompiledModel> {
        let device = CString::new(device)?;
        let mut compiled_model = ptr::null_mut();
        let status = unsafe {
            ov_core_compile_model(
                self.core,
                model.raw,
                device.as_ptr(),
                0,
                &mut compiled_model,
            )
        };
        if status != 0 || compiled_model.is_null() {
            bail!("ov_core_compile_model failed with status {status}");
        }

        Ok(CompiledModel {
            raw: compiled_model,
        })
    }
}

impl Drop for OpenVinoRuntime {
    fn drop(&mut self) {
        if !self.core.is_null() {
            unsafe { ov_core_free(self.core) };
        }
    }
}

unsafe impl Send for OpenVinoRuntime {}
unsafe impl Sync for OpenVinoRuntime {}

pub struct OpenVinoModel {
    raw: *mut ov_model_t,
}

impl Drop for OpenVinoModel {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ov_model_free(self.raw) };
        }
    }
}

pub struct CompiledModel {
    raw: *mut ov_compiled_model_t,
}

impl CompiledModel {
    pub fn output_names(&self) -> Result<Vec<String>> {
        let mut size = 0;
        let status = unsafe { ov_compiled_model_outputs_size(self.raw, &mut size) };
        if status != 0 {
            bail!("ov_compiled_model_outputs_size failed with status {status}");
        }

        let mut names = Vec::with_capacity(size);
        for index in 0..size {
            let mut port = ptr::null_mut();
            let status = unsafe { ov_compiled_model_output_by_index(self.raw, index, &mut port) };
            if status != 0 || port.is_null() {
                bail!("ov_compiled_model_output_by_index({index}) failed with status {status}");
            }

            let mut name = ptr::null_mut();
            let status = unsafe { ov_port_get_any_name(port, &mut name) };
            unsafe { ov_output_const_port_free(port) };
            if status != 0 || name.is_null() {
                bail!("ov_port_get_any_name({index}) failed with status {status}");
            }

            names.push(
                unsafe { CStr::from_ptr(name) }
                    .to_string_lossy()
                    .to_string(),
            );
            unsafe { ov_free(name) };
        }

        Ok(names)
    }

    pub fn infer_i64_inputs_to_f32_output(
        &self,
        inputs: &[(&str, &[i64])],
        input_shape: &[i64],
        output_name: &str,
    ) -> Result<Vec<f32>> {
        let mut infer_request = InferRequest::new(self)?;
        let mut input_tensors = Vec::with_capacity(inputs.len());

        for (name, data) in inputs {
            let tensor = Tensor::from_i64_slice(data, input_shape)?;
            infer_request.set_tensor(name, &tensor)?;
            input_tensors.push(tensor);
        }

        infer_request.infer()?;
        infer_request.get_f32_tensor(output_name)
    }
}

impl Drop for CompiledModel {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ov_compiled_model_free(self.raw) };
        }
    }
}

unsafe impl Send for CompiledModel {}
unsafe impl Sync for CompiledModel {}

struct InferRequest {
    raw: *mut ov_infer_request_t,
}

impl InferRequest {
    fn new(compiled_model: &CompiledModel) -> Result<Self> {
        let mut request = ptr::null_mut();
        let status =
            unsafe { ov_compiled_model_create_infer_request(compiled_model.raw, &mut request) };
        if status != 0 || request.is_null() {
            bail!("ov_compiled_model_create_infer_request failed with status {status}");
        }

        Ok(Self { raw: request })
    }

    fn set_tensor(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        let name = CString::new(name)?;
        let status = unsafe { ov_infer_request_set_tensor(self.raw, name.as_ptr(), tensor.raw) };
        if status != 0 {
            bail!("ov_infer_request_set_tensor failed with status {status}");
        }

        Ok(())
    }

    fn infer(&mut self) -> Result<()> {
        let status = unsafe { ov_infer_request_infer(self.raw) };
        if status != 0 {
            bail!("ov_infer_request_infer failed with status {status}");
        }

        Ok(())
    }

    fn get_f32_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let name = CString::new(name)?;
        let mut tensor = ptr::null_mut();
        let status = unsafe { ov_infer_request_get_tensor(self.raw, name.as_ptr(), &mut tensor) };
        if status != 0 || tensor.is_null() {
            bail!("ov_infer_request_get_tensor failed with status {status}");
        }

        let tensor = Tensor {
            raw: tensor,
            _storage: TensorStorage::None,
        };
        tensor.to_f32_vec()
    }
}

impl Drop for InferRequest {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ov_infer_request_free(self.raw) };
        }
    }
}

struct Tensor {
    raw: *mut ov_tensor_t,
    _storage: TensorStorage,
}

enum TensorStorage {
    None,
    I64 { _data: Vec<i64> },
}

impl Tensor {
    fn from_i64_slice(data: &[i64], shape: &[i64]) -> Result<Self> {
        let expected_len = shape.iter().try_fold(1usize, |acc, dim| {
            let dim = usize::try_from(*dim).map_err(|_| anyhow::anyhow!("negative tensor dim"))?;
            acc.checked_mul(dim)
                .ok_or_else(|| anyhow::anyhow!("tensor shape element count overflow"))
        })?;
        if data.len() != expected_len {
            bail!(
                "tensor data length {} does not match shape element count {expected_len}",
                data.len()
            );
        }

        let mut data = data.to_vec();
        let mut dims = shape.to_vec();
        let shape = ov_shape_t {
            rank: i64::try_from(dims.len())?,
            dims: dims.as_mut_ptr(),
        };
        let mut tensor = ptr::null_mut();
        let status = unsafe {
            ov_tensor_create_from_host_ptr(
                OV_ELEMENT_TYPE_I64,
                shape,
                data.as_mut_ptr().cast::<c_void>(),
                &mut tensor,
            )
        };
        if status != 0 || tensor.is_null() {
            bail!("ov_tensor_create_from_host_ptr failed with status {status}");
        }

        Ok(Self {
            raw: tensor,
            _storage: TensorStorage::I64 { _data: data },
        })
    }

    fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let mut element_type = 0;
        let status = unsafe { ov_tensor_get_element_type(self.raw, &mut element_type) };
        if status != 0 {
            bail!("ov_tensor_get_element_type failed with status {status}");
        }
        if element_type != OV_ELEMENT_TYPE_F32 {
            bail!("expected F32 tensor, got element type {element_type}");
        }

        let mut size = 0;
        let status = unsafe { ov_tensor_get_size(self.raw, &mut size) };
        if status != 0 {
            bail!("ov_tensor_get_size failed with status {status}");
        }

        let mut data = ptr::null_mut();
        let status = unsafe { ov_tensor_data(self.raw, &mut data) };
        if status != 0 || data.is_null() {
            bail!("ov_tensor_data failed with status {status}");
        }

        Ok(unsafe { slice::from_raw_parts(data.cast::<f32>(), size) }.to_vec())
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ov_tensor_free(self.raw) };
        }
    }
}

fn cstring_path(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .map_err(|err| anyhow::anyhow!("path contains interior nul byte: {err}"))
}
