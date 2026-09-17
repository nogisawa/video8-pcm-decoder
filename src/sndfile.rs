use std::ffi::{CStr, CString, c_char, c_int, c_longlong, c_void};
use std::io;
use std::path::Path;

#[repr(C)]
#[derive(Default)]
struct SfInfo {
    frames: c_longlong,
    samplerate: c_int,
    channels: c_int,
    format: c_int,
    sections: c_int,
    seekable: c_int,
}

#[link(name = "sndfile")]
unsafe extern "C" {
    fn sf_open(path: *const c_char, mode: c_int, info: *mut SfInfo) -> *mut c_void;
    fn sf_close(file: *mut c_void) -> c_int;
    fn sf_seek(file: *mut c_void, frames: c_longlong, whence: c_int) -> c_longlong;
    fn sf_readf_float(file: *mut c_void, ptr: *mut f32, frames: c_longlong) -> c_longlong;
    fn sf_strerror(file: *mut c_void) -> *const c_char;
}

pub struct SoundFile {
    handle: *mut c_void,
    pub frames: i64,
    pub sample_rate: u32,
    pub channels: usize,
}

impl SoundFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL in path"))?;
        let mut info = SfInfo::default();
        let handle = unsafe { sf_open(path.as_ptr(), 0x10, &mut info) };
        if handle.is_null() {
            let message = unsafe { CStr::from_ptr(sf_strerror(handle)) }
                .to_string_lossy()
                .into_owned();
            return Err(io::Error::other(message));
        }
        Ok(Self {
            handle,
            frames: info.frames,
            sample_rate: info.samplerate as u32,
            channels: info.channels as usize,
        })
    }

    pub fn read_mono(&mut self, start: i64, frames: usize) -> io::Result<Vec<f32>> {
        if self.channels != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "RF capture must be mono",
            ));
        }
        if unsafe { sf_seek(self.handle, start, 0) } < 0 {
            return Err(io::Error::other("FLAC seek failed"));
        }
        let mut data = vec![0.0_f32; frames];
        let count = unsafe { sf_readf_float(self.handle, data.as_mut_ptr(), frames as i64) };
        if count < 0 {
            return Err(io::Error::other("FLAC read failed"));
        }
        data.truncate(count as usize);
        Ok(data)
    }
}

impl Drop for SoundFile {
    fn drop(&mut self) {
        unsafe {
            sf_close(self.handle);
        }
    }
}
