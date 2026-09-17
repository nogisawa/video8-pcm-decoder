use crate::{BLOCK_BYTES, BLOCKS_PER_FIELD, crc16_video8};

pub const NTSC_BIT_RATE: f64 = 15_734.264 * 368.0;
pub const PAL_BIT_RATE: f64 = 15_625.0 * 368.0;
const PAYLOAD_BITS: usize = 104;
const BLOCK_BITS: usize = 107;

#[derive(Clone)]
pub struct Candidate {
    pub sample: f64,
    pub address: u8,
    pub raw: [u8; BLOCK_BYTES],
}

#[derive(Clone, Copy)]
struct Sos {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

// scipy.signal.butter(3, [1.5e6, 9e6], bandpass, fs=28636360, output=sos).
// Frequencies scale with the configured sample rate, preserving normalized cutoffs.
const SOS: [Sos; 3] = [
    Sos {
        b0: 0.18599736059736707,
        b1: 0.0,
        b2: -0.18599736059736707,
        a1: -0.5756148149044044,
        a2: -0.03741746468563466,
    },
    Sos {
        b0: 1.0,
        b1: 2.0,
        b2: 1.0,
        a1: 0.5179053588630959,
        a2: 0.4425378564077188,
    },
    Sos {
        b0: 1.0,
        b1: -2.0,
        b2: 1.0,
        a1: -1.6542459731282906,
        a2: 0.7560421140578011,
    },
];

fn filter_one(data: &mut [f64], section: Sos) {
    let (mut z1, mut z2) = (0.0, 0.0);
    for value in data {
        let input = *value;
        let output = section.b0 * input + z1;
        z1 = section.b1 * input - section.a1 * output + z2;
        z2 = section.b2 * input - section.a2 * output;
        *value = output;
    }
}

pub fn bandpass_zero_phase(input: &[f32]) -> Vec<f64> {
    let mut data: Vec<f64> = input.iter().map(|&x| x as f64).collect();
    for section in SOS {
        filter_one(&mut data, section);
    }
    data.reverse();
    for section in SOS {
        filter_one(&mut data, section);
    }
    data.reverse();
    data
}

#[inline]
fn interpolate(data: &[f64], position: f64) -> f64 {
    let at = position.floor() as usize;
    if at + 1 >= data.len() {
        return 0.0;
    }
    let fraction = position - at as f64;
    data[at] + (data[at + 1] - data[at]) * fraction
}

fn bits_for_phase(data: &[f64], spb: f64, phase: f64) -> Vec<u8> {
    let count = ((data.len() as f64 - phase) / spb) as usize - 1;
    (0..count)
        .map(|i| {
            let cell = phase + i as f64 * spb;
            let a = interpolate(data, cell + 0.25 * spb);
            let b = interpolate(data, cell + 0.75 * spb);
            (a * b < 0.0) as u8
        })
        .collect()
}

#[inline]
fn pack_byte(bits: &[u8], start: usize) -> u8 {
    let mut value = 0_u8;
    for bit in 0..8 {
        value |= bits[start + bit] << bit;
    }
    value
}

fn scan_phase(filtered: &[f64], absolute_start: i64, spb: f64, phase: f64) -> Vec<Candidate> {
    let bits = bits_for_phase(filtered, spb, phase);
    if bits.len() < PAYLOAD_BITS {
        return Vec::new();
    }
    let mut found = Vec::new();
    for start in 0..=bits.len() - PAYLOAD_BITS {
        // Test the address before packing the remaining 96 bits. Random RF
        // rejects almost half of all candidates here.
        let address = pack_byte(&bits, start);
        if address >= BLOCKS_PER_FIELD as u8 {
            continue;
        }
        let mut raw = [0_u8; BLOCK_BYTES];
        raw[0] = address;
        for byte in 1..BLOCK_BYTES {
            raw[byte] = pack_byte(&bits, start + byte * 8);
        }
        let recorded = u16::from_le_bytes([raw[11], raw[12]]);
        if crc16_video8(&raw[..11]) != recorded {
            continue;
        }
        found.push(Candidate {
            sample: absolute_start as f64 + phase + start as f64 * spb,
            address,
            raw,
        });
    }
    found
}

pub fn scan_window(
    input: &[f32],
    absolute_start: i64,
    sample_rate: f64,
    bit_rate: f64,
    phases: usize,
) -> (Vec<f64>, Vec<Candidate>) {
    scan_window_gated(input, absolute_start, sample_rate, bit_rate, phases, 0.0)
}

/// High-frequency activity relative to the total signal level.  Manchester
/// PCM has many sample-to-sample transitions, whereas an empty part of the
/// capture is comparatively smooth.  The ratio also makes the gate mostly
/// independent of capture gain.
pub fn activity_ratio(input: &[f32]) -> f64 {
    if input.len() < 2 {
        return 0.0;
    }
    let mut signal_energy = 0.0_f64;
    let mut difference_energy = 0.0_f64;
    let mut previous = input[0] as f64;
    signal_energy += previous * previous;
    for &sample in &input[1..] {
        let current = sample as f64;
        let difference = current - previous;
        signal_energy += current * current;
        difference_energy += difference * difference;
        previous = current;
    }
    if signal_energy <= f64::EPSILON {
        return 0.0;
    }
    (difference_energy / signal_energy).sqrt()
}

pub fn scan_window_gated(
    input: &[f32],
    absolute_start: i64,
    sample_rate: f64,
    bit_rate: f64,
    phases: usize,
    gate_threshold: f64,
) -> (Vec<f64>, Vec<Candidate>) {
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(phases);
    scan_window_gated_with_workers(
        input,
        absolute_start,
        sample_rate,
        bit_rate,
        phases,
        gate_threshold,
        workers,
        false,
    )
}

pub fn scan_window_gated_with_workers(
    input: &[f32],
    absolute_start: i64,
    sample_rate: f64,
    bit_rate: f64,
    phases: usize,
    gate_threshold: f64,
    workers: usize,
    stop_after_field_lock: bool,
) -> (Vec<f64>, Vec<Candidate>) {
    if gate_threshold > 0.0 && activity_ratio(input) < gate_threshold {
        return (Vec::new(), Vec::new());
    }
    let filtered = bandpass_zero_phase(input);
    let spb = sample_rate / bit_rate;
    let workers = workers.max(1).min(phases);
    let found = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let filtered = &filtered;
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                let mut phases_after_lock = None::<usize>;
                for phase_index in (worker..phases).step_by(workers) {
                    let phase = phase_index as f64 * spb / phases as f64;
                    local.extend(scan_phase(filtered, absolute_start, spb, phase));
                    if let Some(remaining) = phases_after_lock {
                        // A couple of additional phases supply CRC-valid blocks that
                        // are marginal at the lock phase, while still avoiding
                        // the remaining exhaustive phase scans.
                        if remaining == 1 {
                            break;
                        }
                        phases_after_lock = Some(remaining - 1);
                    }
                    // v8demod only needs enough CRC-valid blocks to locate the
                    // field grid.  Once locked, expected block positions are
                    // decoded directly and CRC decides whether a wider phase
                    // search is necessary.
                    if stop_after_field_lock
                        && workers == 1
                        && phases_after_lock.is_none()
                        && !field_starts(&local, sample_rate, bit_rate).is_empty()
                    {
                        phases_after_lock = Some(2);
                    }
                }
                local
            }));
        }
        let mut combined = Vec::new();
        for handle in handles {
            combined.extend(handle.join().expect("phase worker panicked"));
        }
        combined
    });
    (filtered, found)
}

pub fn field_starts(
    candidates: &[Candidate],
    sample_rate: f64,
    bit_rate: f64,
) -> Vec<(f64, usize)> {
    let step = BLOCK_BITS as f64 * sample_rate / bit_rate;
    let mut estimates: Vec<f64> = candidates
        .iter()
        .map(|c| c.sample - c.address as f64 * step)
        .collect();
    estimates.sort_by(|a, b| a.total_cmp(b));
    let mut result = Vec::new();
    let mut at = 0;
    while at < estimates.len() {
        let mut end = at + 1;
        while end < estimates.len() && estimates[end] - estimates[at] < 6.0 {
            end += 1;
        }
        if end - at >= 6 {
            result.push((estimates[at + (end - at) / 2], end - at));
        }
        at = end;
    }
    result
}

pub fn hard_block_at(
    filtered: &[f64],
    address_sample: f64,
    sample_rate: f64,
    bit_rate: f64,
) -> [u8; BLOCK_BYTES] {
    let spb = sample_rate / bit_rate;
    let mut best = [0_u8; BLOCK_BYTES];
    let mut best_score = -1.0;
    // The field-start estimate comes from several independently decoded
    // blocks and can be displaced by more than half a bit. Search a wide
    // local interval; CRC selects the unambiguous phase.
    // Try the field-grid prediction first.  Most clean blocks finish after
    // this single attempt.  Only a CRC failure expands the search symmetrically
    // out to +/-1.5 bits.
    for trial in 0..49 {
        let step = if trial == 0 {
            0.0
        } else {
            let distance = (trial + 1) / 2;
            let sign = if trial & 1 == 1 { 1.0 } else { -1.0 };
            sign * 1.5 * distance as f64 / 24.0
        };
        let adjust = step * spb;
        let mut raw = [0_u8; BLOCK_BYTES];
        let mut score = 0.0;
        for bit in 0..PAYLOAD_BITS {
            let cell = address_sample + adjust + bit as f64 * spb;
            let a = interpolate(filtered, cell + 0.25 * spb);
            let b = interpolate(filtered, cell + 0.75 * spb);
            raw[bit / 8] |= ((a * b < 0.0) as u8) << (bit % 8);
            score += a.abs().min(b.abs());
        }
        let recorded = u16::from_le_bytes([raw[11], raw[12]]);
        if crc16_video8(&raw[..11]) == recorded {
            return raw;
        }
        if score > best_score {
            best_score = score;
            best = raw;
        }
    }
    best
}

// Kept private to this module; lib.rs has no block-bit constant.
const _: usize = BLOCK_BITS;
