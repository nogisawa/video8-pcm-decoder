use std::io::{self, Read, Write};

pub mod demod;
pub mod sndfile;

pub const BLOCK_BYTES: usize = 13;
pub const BLOCKS_PER_FIELD: usize = 132;
pub const FIELD_BYTES: usize = BLOCK_BYTES * BLOCKS_PER_FIELD;
pub const SAMPLES_PER_FIELD: usize = 525;
pub const WAV_RATE: u32 = 31_469;

/// Stereo reconstruction low-pass used after the 8-to-10-bit expansion.
/// The Video8 PCM passband ends at 15 kHz, very close to the 15.7345 kHz
/// Nyquist frequency.  A real player has an analogue reconstruction filter;
/// raw WAV samples otherwise retain a conspicuous alternating component.
pub struct ReconstructionFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: [f64; 2],
    z2: [f64; 2],
}

impl ReconstructionFilter {
    pub fn new(cutoff_hz: f64) -> Self {
        let omega = 2.0 * std::f64::consts::PI * cutoff_hz / WAV_RATE as f64;
        let (sin, cos) = omega.sin_cos();
        let alpha = sin / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
        let norm = 1.0 / (1.0 + alpha);
        Self {
            b0: (1.0 - cos) * 0.5 * norm,
            b1: (1.0 - cos) * norm,
            b2: (1.0 - cos) * 0.5 * norm,
            a1: -2.0 * cos * norm,
            a2: (1.0 - alpha) * norm,
            z1: [0.0; 2],
            z2: [0.0; 2],
        }
    }

    pub fn process_field(&mut self, audio: &mut [[Option<i16>; 2]; SAMPLES_PER_FIELD]) {
        for frame in audio {
            for (channel, sample) in frame.iter_mut().enumerate() {
                let input = sample.unwrap_or(0) as f64;
                let output = self.b0 * input + self.z1[channel];
                self.z1[channel] = self.b1 * input - self.a1 * output + self.z2[channel];
                self.z2[channel] = self.b2 * input - self.a2 * output;
                *sample = Some(output.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
            }
        }
    }
}

pub fn crc16_video8(data: &[u8]) -> u16 {
    let mut crc = 0xffff_u16;
    for &byte in data {
        crc ^= byte as u16;
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0x8408 } else { 0 };
        }
    }
    crc
}

pub fn crc_ok(block: &[u8]) -> bool {
    let recorded = u16::from_le_bytes([block[11], block[12]]);
    crc16_video8(&block[..11]) == recorded
}

pub fn expand_pcm_byte(value: u8) -> i16 {
    let signed = value as i8 as i16;
    if (-16..=15).contains(&signed) {
        return signed;
    }
    if signed > 0 {
        let (base, width) = if signed < 40 {
            ((signed - 8) * 2, 2)
        } else if signed < 104 {
            ((signed - 24) * 4, 4)
        } else {
            ((signed - 64) * 8, 8)
        };
        return base + width / 2;
    }
    -expand_pcm_byte(((-signed) - 1) as u8)
}

pub fn decode_field(field: &[u8; FIELD_BYTES], use_bad_crc: bool) -> [[Option<i16>; 2]; 525] {
    let mut audio = [[None; 2]; SAMPLES_PER_FIELD];
    let bases = [0_usize, 63, 129, 195, 261, 327, 393, 459];
    // Physical order after address: Q,W0,W1,W2,W3,P,W4,W5,W6,W7.
    let physical_word = [1_usize, 2, 3, 4, 6, 7, 8, 9];
    for address in 0..BLOCKS_PER_FIELD {
        let at = address * BLOCK_BYTES;
        let block = &field[at..at + BLOCK_BYTES];
        if !use_bad_crc && !crc_ok(block) {
            continue;
        }
        let group = address / 44;
        let column = address % 44;
        for word_index in 0..8 {
            let pair = if word_index == 0 {
                if column < 2 {
                    continue;
                }
                (column - 2) / 2
            } else {
                column / 2
            };
            let sample = bases[word_index] + group + 3 * pair;
            let channel = column & 1;
            let byte = block[1 + physical_word[word_index]];
            audio[sample][channel] = Some(expand_pcm_byte(byte));
        }
    }
    audio
}

pub fn conceal_field(audio: &mut [[Option<i16>; 2]; 525]) -> usize {
    let mut missing_total = 0;
    for channel in 0..2 {
        let good: Vec<usize> = (0..SAMPLES_PER_FIELD)
            .filter(|&i| audio[i][channel].is_some())
            .collect();
        missing_total += SAMPLES_PER_FIELD - good.len();
        if good.is_empty() {
            for frame in audio.iter_mut() {
                frame[channel] = Some(0);
            }
            continue;
        }
        for i in 0..SAMPLES_PER_FIELD {
            if audio[i][channel].is_some() {
                continue;
            }
            match good.binary_search(&i) {
                Ok(_) => unreachable!(),
                Err(0) => audio[i][channel] = audio[good[0]][channel],
                Err(pos) if pos == good.len() => {
                    audio[i][channel] = audio[*good.last().unwrap()][channel]
                }
                Err(pos) => {
                    let left = good[pos - 1];
                    let right = good[pos];
                    let a = audio[left][channel].unwrap() as f64;
                    let b = audio[right][channel].unwrap() as f64;
                    let fraction = (i - left) as f64 / (right - left) as f64;
                    // Match NumPy astype(i16), which truncates toward zero.
                    audio[i][channel] = Some((a + (b - a) * fraction) as i16);
                }
            }
        }
    }
    missing_total
}

pub fn write_pcm<W: Write>(audio: &[[Option<i16>; 2]; 525], out: &mut W) -> io::Result<()> {
    let mut bytes = [0_u8; SAMPLES_PER_FIELD * 2 * 2];
    let mut at = 0;
    for frame in audio {
        for sample in frame {
            let pcm = sample.unwrap_or(0).saturating_mul(64);
            bytes[at..at + 2].copy_from_slice(&pcm.to_le_bytes());
            at += 2;
        }
    }
    out.write_all(&bytes)
}

pub fn read_exact_or_eof<R: Read>(reader: &mut R, buffer: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..])? {
            0 if filled == 0 => return Ok(false),
            0 => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "partial Video8 field",
                ));
            }
            n => filled += n,
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_crc() {
        let data = [
            0x00, 0xf8, 0x00, 0x43, 0xb2, 0x32, 0xe4, 0xa8, 0x18, 0xbd, 0xfb,
        ];
        assert_eq!(crc16_video8(&data), 0x922d);
    }

    #[test]
    fn quantizer_centres() {
        assert_eq!(expand_pcm_byte(16), 17);
        assert_eq!(expand_pcm_byte(40), 66);
        assert_eq!(expand_pcm_byte(104), 324);
        assert_eq!(expand_pcm_byte(128), -508);
    }
}
